document.addEventListener("DOMContentLoaded", function () {
    // Handling the file upload functionality
    const fileInput = document.getElementById('file');
    const uploadButton = document.querySelector("form button[type='submit']");
    const predictionButton = document.querySelector("form[action='/predict'] button[type='submit']");
    const heatmapButton = document.querySelector("form[action='/generate_heatmap'] button[type='submit']");
    const alertContainer = document.querySelector('.alert');
    
    // Handle form submission to upload the image
    if (uploadButton) {
        uploadButton.addEventListener('click', function (event) {
            // Make sure a file is selected before submitting the form
            if (!fileInput.files.length) {
                event.preventDefault();
                showAlert('Please select an image to upload.', 'danger');
            }
        });
    }

    // Handle prediction button click
    if (predictionButton) {
        predictionButton.addEventListener('click', function (event) {
            // Add any validation if needed before prediction
            showAlert('Prediction is being processed...', 'info');
        });
    }

    // Handle heatmap generation button click
    if (heatmapButton) {
        heatmapButton.addEventListener('click', function (event) {
            showAlert('Generating heatmap...', 'info');
        });
    }

    // Function to show alert messages (if any)
    function showAlert(message, type) {
        if (alertContainer) {
            alertContainer.innerHTML = `<ul><li>${message}</li></ul>`;
            alertContainer.classList.add(`alert-${type}`);
            alertContainer.style.display = 'block';
        }
    }

    // Close the alert message after 5 seconds
    setTimeout(() => {
        if (alertContainer) {
            alertContainer.style.display = 'none';
        }
    }, 5000);

    // Handle the file input change event for custom image preview if needed
    if (fileInput) {
        fileInput.addEventListener('change', function () {
            const file = fileInput.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function (e) {
                    // Add any image preview functionality here
                    console.log("Image selected: ", file.name);
                };
                reader.readAsDataURL(file);
            }
        });
    }
});
