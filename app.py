import sys
import os
import datetime
import traceback
import uuid 
import threading 
import time 

# --- Add directory to path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# --- End path addition ---

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
from werkzeug.utils import secure_filename
from analysis import run_analysis_with_progress

# --- Configuration ---
FILES_FOLDER = os.path.join(current_dir, 'files') # Stores timestamped uploads
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = FILES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure upload ('files') folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- In-memory storage for task status and results ---
# Use Redis or connect to local datawarehouse for production.
task_status = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_analysis_background(task_id, filepath, original_filename):
    """Wrapper to run analysis and store results/status."""
    print(f"[{task_id}] Starting analysis thread for: {filepath}")
    try:
        # Call the modified analysis function which updates task_status
        run_analysis_with_progress(task_id, filepath, task_status)
        # Status will be updated to 'Complete' or 'Failed' inside the function
        if task_status[task_id]['status'] == 'Complete':
             print(f"[{task_id}] Analysis thread completed successfully.")
             # Store original filename for context if needed later
             task_status[task_id]['original_filename'] = original_filename
        else:
             print(f"[{task_id}] Analysis thread finished with status: {task_status[task_id]['status']}")

    except Exception as e:
        print(f"[{task_id}] Error in analysis thread: {e}")
        traceback.print_exc()
        task_status[task_id] = {
            'status': 'Failed',
            'progress': 100, # Indicate process finished (failed)
            'error': f"An internal error occurred during analysis: {str(e)}"
        }


@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html') # Simpler, results loaded via JS

@app.route('/upload', methods=['POST'])
def upload_file_async():
    """Handles file upload, saves it, starts background analysis, returns task ID."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    original_filename = file.filename

    if original_filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(original_filename):
        # Generate new filename based on datetime
        now = datetime.datetime.now()
        new_filename = now.strftime('%Y%m%d_%H%M%S') + ".csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

        try:
            file.save(filepath)
            print(f"File saved: {filepath}")

            # Generate task ID and initialize status
            task_id = str(uuid.uuid4())
            task_status[task_id] = {'status': 'Pending', 'progress': 0, 'filename': new_filename}

            # Start analysis in a background thread
            thread = threading.Thread(target=run_analysis_background, args=(task_id, filepath, original_filename))
            thread.start()

            print(f"[{task_id}] Analysis thread started.")
            # Return task ID immediately so client can poll
            return jsonify({'task_id': task_id, 'filename': new_filename})

        except Exception as e:
            print(f"Error during file save or thread start: {e}")
            traceback.print_exc()
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

@app.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    """Provides progress updates for a given task."""
    status = task_status.get(task_id, {'status': 'Unknown', 'progress': 0})
    # print(f"Polling progress for {task_id}: {status}") # Can be noisy
    return jsonify(status)

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    """Provides the final analysis result when complete."""
    result = task_status.get(task_id)
    if result and result.get('status') == 'Complete':
        # Return relevant result data (plot, table, r2, filename)
        return jsonify({
            'status': 'Complete',
            'plot_url': result.get('plot_url'),
            'data_html': result.get('data_html'),
            'r2_score': f"{result.get('r2'):.4f}" if result.get('r2') is not None else "N/A",
            'filename': result.get('filename'), # The timestamped filename
            'original_filename': result.get('original_filename') # Optional original name
        })
    elif result and result.get('status') == 'Failed':
         return jsonify({
            'status': 'Failed',
            'error': result.get('error', 'Analysis failed. Check logs.'),
            'filename': result.get('filename'),
         }), 400 # Use a 400 or 500 status for client-side error handling
    elif result:
         # Still processing or pending
         return jsonify({'status': result.get('status', 'Processing'), 'progress': result.get('progress', 0)}), 202 # Accepted, but not ready
    else:
        return jsonify({'status': 'Unknown', 'error': 'Task ID not found.'}), 404


@app.route('/files', methods=['GET'])
def list_files():
    """Lists uploaded files from the 'files' directory, sorted by date."""
    file_list = []
    try:
        filenames = os.listdir(app.config['UPLOAD_FOLDER'])
        csv_files = [f for f in filenames if f.lower().endswith('.csv')]

        for filename in csv_files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                mod_time_epoch = os.path.getmtime(filepath)
                mod_time = datetime.datetime.fromtimestamp(mod_time_epoch)
                # Attempt to parse datetime from filename for sorting consistency
                try:
                    file_dt = datetime.datetime.strptime(filename.split('.')[0], '%Y%m%d_%H%M%S')
                    sort_key = file_dt
                except ValueError:
                    sort_key = mod_time # Fallback to modification time

                file_list.append({'name': filename, 'mod_time': mod_time, 'sort_key': sort_key})
            except OSError:
                print(f"Warning: Could not access file info for {filename}")
                continue

        # Sort by the parsed datetime from filename or mod_time, newest first
        file_list.sort(key=lambda x: x['sort_key'], reverse=True)

    except FileNotFoundError:
        flash(f"Storage directory '{app.config['UPLOAD_FOLDER']}' not found.", 'danger') # Flash might not show if redirected
        print(f"Error: Storage directory '{app.config['UPLOAD_FOLDER']}' not found.")
    except Exception as e:
        flash(f"Error listing files: {e}", 'danger')
        print(f"Error listing files: {e}")
        traceback.print_exc()

    return render_template('files.html', files=file_list)


if __name__ == '__main__':
    # Use host='0.0.0.0' to make accessible on your local network
    # Use threaded=True if using Flask's dev server with background threads (less critical with separate threads)
    app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)