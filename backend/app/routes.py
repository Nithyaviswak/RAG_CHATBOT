import logging
import os
import uuid
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from werkzeug.utils import secure_filename

from .chat_service import ChatService
from .document_service import DocumentService

api = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.doc', '.docx'}

_doc_service = None
_chat_service = None


@api.after_request
def add_cors(response):
    origin = request.headers.get('Origin', '')
    allowed = current_app.config.get('CORS_ORIGINS', [])
    if origin in allowed or '*' in allowed:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response


@api.route('/<path:dummy>', methods=['OPTIONS'])
def preflight(dummy):
    return '', 204


def get_document_service():
    global _doc_service
    if _doc_service is None:
        _doc_service = DocumentService(current_app.config)
    return _doc_service


def get_chat_service():
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService(current_app.config)
    return _chat_service


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _chat_retrieval_mode() -> str:
    mode = (current_app.config.get('CHAT_RETRIEVAL_MODE') or 'vector').strip().lower()
    return mode if mode in {'vector', 'long_context', 'hybrid'} else 'vector'


@api.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413


@api.route('/health', methods=['GET', 'OPTIONS'])
def health():
    return jsonify({'status': 'ok'})


@api.route('/upload', methods=['POST', 'OPTIONS'])
def upload_document():
    if request.method == 'OPTIONS':
        return '', 204

    doc_service = get_document_service()

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Use field name: "file"'}), 400

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}'}), 400

    filename = secure_filename(file.filename)
    temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f'tmp_{uuid.uuid4().hex}_{filename}')

    try:
        file.save(temp_path)

        dup_id = doc_service.is_duplicate(temp_path)
        if dup_id:
            os.remove(temp_path)
            existing = doc_service.get_document(dup_id)
            return jsonify({'error': f'Already uploaded as "{existing["filename"]}"', 'existing_id': dup_id}), 409

        doc_id, meta = doc_service.save_document(temp_path, filename)

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.exception('Save error')
        return jsonify({'error': f'Could not save file: {str(e)}'}), 500

    mode = _chat_retrieval_mode()
    if mode == 'long_context':
        try:
            meta = doc_service.mark_document_ready_for_long_context(doc_id)
            return jsonify({
                'message': 'Document uploaded and prepared for long-context chat',
                'document': {k: v for k, v in meta.items() if k != 'hash'},
            }), 201
        except Exception as e:
            logger.exception('Long-context preparation error')
            return jsonify({
                'message': 'File saved but preparation failed.',
                'document': {k: v for k, v in meta.items() if k != 'hash'},
                'error': str(e),
            }), 202

    try:
        meta = doc_service.vectorise_document(doc_id)
        return jsonify({
            'message': 'Document uploaded and processed successfully',
            'document': {k: v for k, v in meta.items() if k != 'hash'},
        }), 201

    except RuntimeError as e:
        logger.warning(f'Vectorisation skipped: {e}')
        return jsonify({
            'message': 'File saved but processing is pending. Check embedding configuration and retry.',
            'document': {k: v for k, v in meta.items() if k != 'hash'},
            'warning': str(e),
        }), 202

    except Exception as e:
        logger.exception('Vectorisation error')
        return jsonify({
            'message': 'File saved but processing failed. Check your API key.',
            'document': {k: v for k, v in meta.items() if k != 'hash'},
            'error': str(e),
        }), 202


@api.route('/documents', methods=['GET', 'OPTIONS'])
def list_documents():
    if request.method == 'OPTIONS':
        return '', 204
    doc_service = get_document_service()
    return jsonify({'documents': doc_service.list_documents()})


@api.route('/document/<doc_id>', methods=['DELETE', 'OPTIONS'])
def delete_document(doc_id):
    if request.method == 'OPTIONS':
        return '', 204
    doc_service = get_document_service()
    success = doc_service.delete_document(doc_id)
    if success:
        return jsonify({'message': 'Document deleted successfully'})
    return jsonify({'error': 'Document not found'}), 404


@api.route('/document/<doc_id>/process', methods=['POST', 'OPTIONS'])
def reprocess_document(doc_id):
    if request.method == 'OPTIONS':
        return '', 204

    doc_service = get_document_service()
    meta = doc_service.get_document(doc_id)
    if not meta:
        return jsonify({'error': 'Document not found'}), 404

    mode = _chat_retrieval_mode()
    try:
        if mode == 'long_context':
            meta = doc_service.mark_document_ready_for_long_context(doc_id)
            return jsonify({
                'message': 'Document prepared for long-context chat',
                'document': {k: v for k, v in meta.items() if k != 'hash'},
            })

        meta = doc_service.vectorise_document(doc_id)
        return jsonify({
            'message': 'Document processed successfully',
            'document': {k: v for k, v in meta.items() if k != 'hash'},
        })
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception('Reprocess error')
        return jsonify({'error': str(e)}), 500


@api.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 204

    doc_service = get_document_service()
    chat_service = get_chat_service()
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    question = data.get('question', '').strip()
    doc_ids = data.get('doc_ids', [])
    session_id = data.get('session_id') or str(uuid.uuid4())
    do_stream = data.get('stream', False)

    if not question:
        return jsonify({'error': 'Question is required'}), 400
    if not doc_ids:
        return jsonify({'error': 'Select at least one document'}), 400

    mode = _chat_retrieval_mode()
    use_long_context = mode == 'long_context'

    vectorstore = None
    if mode in {'vector', 'hybrid'}:
        vectorstore = (
            doc_service.get_vectorstore(doc_ids[0])
            if len(doc_ids) == 1
            else doc_service.get_combined_vectorstore(doc_ids)
        )
        if vectorstore is None and mode == 'hybrid':
            use_long_context = True

    if vectorstore is None and not use_long_context:
        return jsonify({
            'error': (
                'Selected document(s) are not ready for vector retrieval. '
                'Reprocess documents or set CHAT_RETRIEVAL_MODE=long_context/hybrid.'
            )
        }), 404

    long_context = ''
    long_sources = []
    if use_long_context:
        max_chars = int(current_app.config.get('LONG_CONTEXT_MAX_CHARS', 300000))
        long_context, long_sources = doc_service.get_long_context(doc_ids, max_chars=max_chars)
        if not long_context.strip():
            return jsonify({'error': 'Could not read selected document content for long-context chat.'}), 404

    if do_stream:
        def generate():
            if use_long_context:
                for chunk in chat_service.chat_stream_with_context(session_id, question, long_context, long_sources):
                    yield chunk
                return
            for chunk in chat_service.chat_stream(session_id, question, vectorstore):
                yield chunk

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
        )

    try:
        if use_long_context:
            result = chat_service.chat_with_context(session_id, question, long_context, long_sources)
        else:
            result = chat_service.chat(session_id, question, vectorstore)
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception('Chat error')
        return jsonify({'error': str(e)}), 500


@api.route('/reset-memory', methods=['POST', 'OPTIONS'])
def reset_memory():
    if request.method == 'OPTIONS':
        return '', 204
    chat_service = get_chat_service()
    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'session_id is required'}), 400
    chat_service.reset_session(session_id)
    return jsonify({'message': 'Memory reset successfully', 'session_id': session_id})


@api.route('/history', methods=['GET', 'OPTIONS'])
def get_history():
    if request.method == 'OPTIONS':
        return '', 204
    chat_service = get_chat_service()
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'session_id query param required'}), 400
    history = chat_service.get_history(session_id)
    return jsonify({'history': history, 'session_id': session_id})
