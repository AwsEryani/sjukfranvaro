// static/js/script.js - Modified with Debug Logs

document.addEventListener('DOMContentLoaded', function() {
    console.log("[Debug] DOMContentLoaded event fired. Initializing script..."); // Debug Log

    // --- Get DOM Elements ---
    const uploadForm = document.getElementById('upload-form');
    const uploadButton = document.getElementById('upload-button');
    const uploadSpinner = document.getElementById('upload-spinner');
    const fileInput = document.getElementById('file');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressStatus = document.getElementById('progress-status');
    const progressFilename = document.getElementById('progress-filename');
    const resultsArea = document.getElementById('results-area');
    const resultsFilename = document.getElementById('results-filename');
    const r2ScoreArea = document.getElementById('r2-score-area');
    const r2ScoreValue = document.getElementById('r2-score-value');
    const analysisErrorArea = document.getElementById('analysis-error-area');
    const analysisErrorMessage = document.getElementById('analysis-error-message');
    const plotCard = document.getElementById('plot-card');
    const resultPlot = document.getElementById('result-plot');
    const tableCard = document.getElementById('table-card');
    const resultTableDiv = document.getElementById('result-table');
    const loadMoreBtn = document.getElementById('load-more-btn');
    const tableTestYearLabel = document.getElementById('table-test-year-label');

    // --- Check if essential elements exist ---
    if (!uploadForm) {
        console.error("[Debug] CRITICAL: Element with ID 'upload-form' not found!");
        return; // Stop script execution if form is missing
    }
     if (!uploadButton) console.warn("[Debug] Element with ID 'upload-button' not found.");
     if (!fileInput) console.warn("[Debug] Element with ID 'file' not found.");
     if (!progressContainer) console.warn("[Debug] Element with ID 'progress-container' not found.");
     if (!resultsArea) console.warn("[Debug] Element with ID 'results-area' not found.");
     // Add more checks if needed


    // --- Global Variables ---
    let pollInterval; // To store the interval ID for polling
    const ROWS_PER_PAGE = 10; // How many rows to show per "Ladda Mer" click
    let currentVisibleRows = 0; // Track how many rows are currently shown

    console.log("[Debug] DOM elements obtained. Adding submit listener..."); // Debug Log

    // --- Form Submission ---
    uploadForm.addEventListener('submit', function(event) {
        console.log("[Debug] 'submit' event triggered on #upload-form."); // Debug Log

        event.preventDefault(); // Prevent default form submission

        // Basic check if a file is selected
        if (!fileInput.files || fileInput.files.length === 0) {
            console.warn("[Debug] No file selected."); // Debug Log
            showFlashMessage('Please select a file to upload.', 'warning');
            return;
        }
        console.log("[Debug] File selected:", fileInput.files[0].name); // Debug Log

        // Disable button, show spinner
        uploadButton.disabled = true;
        uploadSpinner.style.display = 'inline-block';
        console.log("[Debug] Upload button disabled, spinner shown."); // Debug Log
        hideResultsAndProgress(); // Hide previous results/progress

        const formData = new FormData(uploadForm);
        console.log("[Debug] FormData created."); // Debug Log

        console.log("[Debug] Initiating fetch request to:", uploadForm.action); // Debug Log
        // Use fetch to submit the form asynchronously
        fetch(uploadForm.action, {
                method: 'POST',
                body: formData,
            })
            .then(response => {
                console.log("[Debug] Fetch response received. Status:", response.status); // Debug Log
                if (!response.ok) {
                    console.warn("[Debug] Fetch response not OK. Status:", response.status); // Debug Log
                    // Try to parse error message from JSON response
                    return response.json().then(err => {
                         console.error("[Debug] Parsed error JSON from server:", err); // Debug Log
                        // Throw error to be caught by .catch()
                        throw new Error(err.error || `HTTP error! Status: ${response.status}`);
                    }).catch(jsonParseError => {
                        // Handle cases where the response wasn't even JSON
                        console.error("[Debug] Response was not JSON or JSON parsing failed:", jsonParseError);
                        throw new Error(`HTTP error! Status: ${response.status}. Server response was not valid JSON.`);
                    });
                }
                console.log("[Debug] Fetch response OK. Parsing JSON..."); // Debug Log
                return response.json();
            })
            .then(data => {
                console.log("[Debug] Parsed JSON data from fetch:", data); // Debug Log
                if (data.task_id) {
                    console.log("[Debug] Task ID received:", data.task_id, "Filename:", data.filename); // Debug Log
                    progressFilename.textContent = `"${data.filename}"`; // Display the saved filename
                    progressContainer.style.display = 'block'; // Show progress bar
                    resetProgressBar();
                    startPolling(data.task_id); // Start polling for progress
                } else {
                     console.error("[Debug] Task ID missing in successful response:", data); // Debug Log
                    // Should not happen if response.ok, but handle defensively
                    throw new Error(data.error || 'Invalid response from server (missing task_id).');
                }
            })
            .catch(error => {
                // This catches errors from fetch(), network issues, or errors thrown above
                console.error('[Debug] Error during upload fetch/processing:', error); // Debug Log
                showFlashMessage(`Upload failed: ${error.message}`, 'danger');
                // Re-enable button, hide spinner
                uploadButton.disabled = false;
                uploadSpinner.style.display = 'none';
                console.log("[Debug] Upload button re-enabled, spinner hidden due to error."); // Debug Log
            });
         console.log("[Debug] Fetch call initiated (asynchronous)."); // Debug Log
    });
    console.log("[Debug] Submit event listener attached successfully."); // Debug Log


    // --- Progress Polling ---
    function startPolling(taskId) {
        console.log(`[Debug] Starting polling for task ID: ${taskId}`); // Debug Log
        pollInterval = setInterval(() => {
            console.log(`[Debug] Polling progress for ${taskId}...`); // Debug Log (can be noisy)
            fetch(`/progress/${taskId}`)
                .then(response => {
                     console.log(`[Debug] Poll response status for ${taskId}: ${response.status}`); // Debug Log
                    if (!response.ok) {
                        throw new Error(`Progress check failed! Status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                     console.log(`[Debug] Poll data for ${taskId}:`, data); // Debug Log
                    updateProgressBar(data.progress, data.status);

                    if (data.status === 'Complete' || data.status === 'Failed') {
                         console.log(`[Debug] Polling stopped for ${taskId}. Status: ${data.status}`); // Debug Log
                        clearInterval(pollInterval); // Stop polling
                        pollInterval = null; // Clear interval ID
                        fetchResults(taskId); // Fetch final results or error message
                    }
                })
                .catch(error => {
                    console.error(`[Debug] Polling error for ${taskId}:`, error); // Debug Log
                    updateProgressBar(100, 'Error'); // Mark as error
                    showFlashMessage(`Error checking progress: ${error.message}`, 'danger');
                    if (pollInterval) clearInterval(pollInterval); // Ensure polling stops on error
                    pollInterval = null;
                    // Re-enable button, hide spinner
                    uploadButton.disabled = false;
                    uploadSpinner.style.display = 'none';
                     // Hide progress bar on error after a delay
                    setTimeout(() => { progressContainer.style.display = 'none'; }, 3000);
                });
        }, 2000); // Poll every 2 seconds
    }

    // --- Fetch Final Results ---
    function fetchResults(taskId) {
        console.log(`[Debug] Fetching final results for task ID: ${taskId}`); // Debug Log
        fetch(`/result/${taskId}`)
            .then(response => {
                 console.log(`[Debug] Result fetch response status for ${taskId}: ${response.status}`); // Debug Log
                 if (!response.ok) {
                    return response.json().then(errData => {
                         console.error(`[Debug] Error response JSON from /result/${taskId}:`, errData); // Debug Log
                         throw {
                             status: 'Failed',
                             error: errData.error || `Failed to fetch results (Status: ${response.status})`,
                             filename: errData.filename || 'Unknown file'
                         };
                    }).catch(jsonParseError => {
                         console.error(`[Debug] Result response for ${taskId} not JSON or parse failed:`, jsonParseError);
                         throw new Error(`Failed to fetch results (Status: ${response.status}). Server response was not valid JSON.`);
                    });
                 }
                 return response.json();
            })
            .then(data => {
                 console.log(`[Debug] Result data received for ${taskId}:`, data); // Debug Log
                 progressContainer.style.display = 'none';
                 uploadButton.disabled = false;
                 uploadSpinner.style.display = 'none';
                 console.log("[Debug] Progress hidden, upload button enabled after fetching results."); // Debug Log

                if (data.status === 'Complete') {
                     console.log(`[Debug] Task ${taskId} complete. Displaying results.`); // Debug Log
                    displayResults(data);
                } else if (data.status === 'Failed') {
                     console.error(`[Debug] Task ${taskId} failed. Displaying error:`, data.error); // Debug Log
                    displayError(data.error || 'Unknown analysis failure.', data.filename);
                } else {
                    console.warn(`[Debug] Unexpected status '${data.status}' received from /result/${taskId}.`); // Debug Log
                    displayError(`Unexpected status: ${data.status}. Please try again.`, data.filename || 'Unknown file');
                }
            })
            .catch(error => {
                 console.error(`[Debug] Error fetching/processing final result for ${taskId}:`, error); // Debug Log
                 if (error && error.status === 'Failed'){
                     displayError(error.error, error.filename);
                 } else {
                    displayError(`Failed to load results: ${error.message}`, 'Error');
                 }
                 // Ensure UI is reset
                 progressContainer.style.display = 'none';
                 uploadButton.disabled = false;
                 uploadSpinner.style.display = 'none';
                 console.log("[Debug] UI reset due to result fetch error."); // Debug Log
            });
    }

    // --- UI Update Functions ---

    function updateProgressBar(percent, statusText) {
        percent = Math.max(0, Math.min(100, percent || 0));
        progressBar.style.width = percent + '%';
        progressBar.setAttribute('aria-valuenow', percent);
        progressBar.textContent = percent + '%';
        progressStatus.textContent = statusText || 'Processing...';
        // Color change logic remains the same
        progressBar.classList.remove('bg-success', 'bg-danger', 'bg-gold');
        if (statusText === 'Complete') progressBar.classList.add('bg-success');
        else if (statusText === 'Failed' || statusText === 'Error') progressBar.classList.add('bg-danger');
        else progressBar.classList.add('bg-gold');
    }

     function resetProgressBar() {
         progressBar.style.width = '0%';
         progressBar.setAttribute('aria-valuenow', 0);
         progressBar.textContent = '0%';
         progressStatus.textContent = 'Starting...';
         progressBar.classList.remove('bg-success', 'bg-danger');
         progressBar.classList.add('bg-gold');
         console.log("[Debug] Progress bar reset."); // Debug Log
     }

    function displayResults(data) {
        console.log("[Debug] Displaying results:", data); // Debug Log
        hideResultsAndProgress(); // Clear previous state first
        resultsArea.style.display = 'block';
        resultsFilename.textContent = `"${data.filename}"`;

        // R2 Score Display Logic (no change needed)
        if (data.r2_score && data.r2_score !== "N/A" && data.r2_score !== "Error") {
            r2ScoreValue.textContent = data.r2_score;
            r2ScoreValue.className = 'badge fs-6 bg-success gold-text';
            r2ScoreArea.style.display = 'block';
        } else if (data.r2_score === "Error") {
            r2ScoreValue.textContent = "Calculation Failed";
            r2ScoreValue.className = 'badge fs-6 bg-danger';
            r2ScoreArea.style.display = 'block';
        } else {
            r2ScoreArea.style.display = 'none';
        }

        // Plot Display Logic (no change needed)
        if (data.plot_url) {
            resultPlot.src = "data:image/png;base64," + data.plot_url;
            plotCard.style.display = 'block';
        } else {
            plotCard.style.display = 'none';
        }

        // Table Display Logic
        if (data.data_html) {
            resultTableDiv.innerHTML = data.data_html;
            tableCard.style.display = 'block';
            // Test Year Label Logic (no change needed)
            const tableElement = resultTableDiv.querySelector('table'); // Use querySelector on the div now
            if (tableElement && tableElement.tHead && tableElement.tHead.rows.length > 0) {
                const headers = Array.from(tableElement.tHead.rows[0].cells).map(cell => cell.textContent);
                const testSetIndex = headers.indexOf('Test Set');
                if (testSetIndex > -1 && tableElement.tBodies[0] && tableElement.tBodies[0].rows.length > 0) {
                   // Check if cell exists before accessing textContent
                   const cell = tableElement.tBodies[0].rows[0].cells[testSetIndex];
                   tableTestYearLabel.textContent = cell ? `(Test Set: ${cell.textContent})` : '';
                } else {
                    tableTestYearLabel.textContent = '';
                }
            } else {
                 tableTestYearLabel.textContent = '';
            }

            console.log("[Debug] Setting up table pagination."); // Debug Log
            setupTablePagination(); // Initialize pagination for the new table
        } else {
            console.log("[Debug] No table HTML received."); // Debug Log
            tableCard.style.display = 'none';
            resultTableDiv.innerHTML = ''; // Clear previous table
        }
         console.log("[Debug] displayResults function finished."); // Debug Log
    }

    function displayError(errorMessage, filename) {
         console.error(`[Debug] Displaying error: ${errorMessage} (File: ${filename || 'N/A'})`); // Debug Log
         hideResultsAndProgress(); // Clear previous state
         resultsArea.style.display = 'block';
         resultsFilename.textContent = filename ? `"${filename}"` : "Analysis";
         analysisErrorMessage.textContent = errorMessage;
         analysisErrorArea.style.display = 'block';
         // Hide other result parts
         r2ScoreArea.style.display = 'none';
         plotCard.style.display = 'none';
         tableCard.style.display = 'none';
    }

    function hideResultsAndProgress() {
        console.log("[Debug] Hiding results and progress areas."); // Debug Log
        if (pollInterval) {
            console.log("[Debug] Clearing active polling interval."); // Debug Log
            clearInterval(pollInterval);
            pollInterval = null;
        }
        progressContainer.style.display = 'none';
        resultsArea.style.display = 'none';
        analysisErrorArea.style.display = 'none';
        r2ScoreArea.style.display = 'none';
        plotCard.style.display = 'none';
        tableCard.style.display = 'none';
        resultTableDiv.innerHTML = '';
        loadMoreBtn.style.display = 'none';
        currentVisibleRows = 0;
    }

    // --- Table Pagination ---
    function setupTablePagination() {
        const table = resultTableDiv.querySelector('table.results-table-data');
        if (!table || !table.tBodies || table.tBodies.length === 0) {
            console.warn("[Debug] No table body found for pagination setup."); // Debug Log
            loadMoreBtn.style.display = 'none';
            return;
        }
        const tbody = table.tBodies[0];
        const allRows = Array.from(tbody.rows);
        const totalRows = allRows.length;
        console.log(`[Debug] Found ${totalRows} rows for pagination.`); // Debug Log

        currentVisibleRows = 0;

        allRows.forEach(row => row.style.display = 'none');
        console.log("[Debug] All table rows hidden initially."); // Debug Log

        showNextRows(allRows, totalRows); // Show the first batch

        // Setup Ladda Mer button listener
        loadMoreBtn.removeEventListener('click', handleLoadMore); // Ensure no duplicates
        loadMoreBtn.addEventListener('click', handleLoadMore);
        console.log("[Debug] 'Ladda Mer' button listener attached."); // Debug Log
    }

    // Named function for the event listener
     function handleLoadMore() {
         console.log("[Debug] 'Ladda Mer' clicked."); // Debug Log
         const table = resultTableDiv.querySelector('table.results-table-data');
         if (!table || !table.tBodies || table.tBodies.length === 0) return;
         const tbody = table.tBodies[0];
         const allRows = Array.from(tbody.rows);
         const totalRows = allRows.length;
         showNextRows(allRows, totalRows);
     }


    function showNextRows(allRows, totalRows) {
        const start = currentVisibleRows;
        const end = currentVisibleRows + ROWS_PER_PAGE;
        const rowsToShow = allRows.slice(start, end);
        console.log(`[Debug] Showing rows ${start + 1} to ${Math.min(end, totalRows)} of ${totalRows}.`); // Debug Log
        rowsToShow.forEach(row => row.style.display = ''); // Show rows

        currentVisibleRows += rowsToShow.length;

        // Update Ladda Mer button visibility
        if (currentVisibleRows >= totalRows) {
            loadMoreBtn.style.display = 'none';
            console.log("[Debug] All rows shown, hiding 'Ladda Mer' button."); // Debug Log
        } else {
            loadMoreBtn.style.display = 'inline-block';
            loadMoreBtn.textContent = `Ladda Mer (${currentVisibleRows} / ${totalRows})`;
            console.log("[Debug] More rows available, showing 'Ladda Mer' button."); // Debug Log
        }
    }

     // --- Flash Message Simulation ---
     function showFlashMessage(message, category = 'info') {
         // Use the dedicated container now in base.html
         const flashContainer = document.getElementById('flash-container'); // Use ID now
         if (!flashContainer) {
             console.error("[Debug] Flash container '#flash-container' not found!");
             return;
         }
          console.log(`[Debug] Showing flash message: "${message}" (Category: ${category})`); // Debug Log

         const alertDiv = document.createElement('div');
         const alertClass = `alert-${category}`;
         alertDiv.className = `alert ${alertClass} alert-dismissible fade show`;
         alertDiv.setAttribute('role', 'alert');

          let iconClass = 'fa-info-circle';
          if (category === 'success') iconClass = 'fa-check-circle';
          else if (category === 'danger') iconClass = 'fa-exclamation-triangle';
          else if (category === 'warning') iconClass = 'fa-exclamation-circle';

         alertDiv.innerHTML = `
             <i class="fas ${iconClass} me-2"></i>
             ${message}
             <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
         `;

         // Prepend to the container
         flashContainer.insertBefore(alertDiv, flashContainer.firstChild);

         // Auto-dismiss after 5 seconds (optional)
         setTimeout(() => {
             // Use Bootstrap's Alert instance to close smoothly
             const alertInstance = bootstrap.Alert.getOrCreateInstance(alertDiv);
             if (alertInstance) {
                 alertInstance.close();
             } else {
                 // Fallback if Bootstrap JS isn't fully loaded yet or fails
                  alertDiv.remove();
             }
         }, 5000);
     }


}); // End DOMContentLoaded

console.log("Custom script loaded. DOMContentLoaded listener is waiting."); // Debug Log