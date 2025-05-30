// Handle image upload and display the selected image
document.getElementById('imageUpload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const fileLabel = document.getElementById('fileLabel');
    const uploadedImage = document.getElementById('uploadedImage');

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadedImage.src = e.target.result; // Set the image source to the uploaded file
            uploadedImage.style.display = 'block'; // Show the uploaded image
            fileLabel.style.display = 'none'; // Hide the file label
        }
        reader.readAsDataURL(file); // Read the file as a data URL
    } else {
        // If no file is selected, hide the image and show the label
        uploadedImage.style.display = 'none';
        fileLabel.style.display = 'block';
    }
});

// Handle skin analysis form submission
document.getElementById('skinAnalysisForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    const imageUpload = document.getElementById('imageUpload').files[0];
    const formData = new FormData();
    formData.append('image', imageUpload);

    // Send the image to the server for analysis
    fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Check if data is an object and has properties
        if (data && typeof data === 'object') {
            // Display skin metrics
            document.getElementById('skinMetrics').innerHTML = '';
            const metrics = [ 
                `Skin Tone: ${Array.isArray(data.tone) ? data.tone.join(', ') : '0%'}`,
                `Acne Level: ${Array.isArray(data.acne_level) ? data.acne_level.join(', ') : '0%'}`,
                `Blackheads: ${Array.isArray(data.blackheads) ? data.blackheads.join(', ') : '0%'}`,
                `Dark Circles: ${Array.isArray(data.dark_circles) ? data.dark_circles.join(', ') : '0%'}`,
                `Skin Type: ${Array.isArray(data.skin_type) ? data.skin_type.join(', ') : '0%'}`,
                `Hair Quality: ${Array.isArray(data.hair_quality) ? data.hair_quality.join(', ') : '0%'}`,
                `Hydration Level: ${Array.isArray(data.hydration_level) ? data.hydration_level.join(', ') : '0%'}`,
                `Sensitivity: ${Array.isArray(data.sensitivity) ? data.sensitivity.join(', ') : '0%'}`,
                `Wrinkles: ${Array.isArray(data.wrinkles) ? data.wrinkles.join(', ') : '0%'}`,
                `Pore Size: ${Array.isArray(data.pore_size) ? data.pore_size.join(', ') : '0%'}`
            ];
           
            metrics.forEach(metric => {
                const li = document.createElement('li');
                li.textContent = metric; // Set the text for the list item
                document.getElementById('skinMetrics').appendChild(li); // Append to the skin metrics list
            });

            // Check if recommendedProducts exists and is an array
            const recommendations = Array.isArray(data.recommendedProducts) && data.recommendedProducts.length > 0 ? 
                data.recommendedProducts.map(product => `<li>${product}</li>`).join('') : 
                '<li>No recommendations available</li>';
            
            document.getElementById('productRecommendations').innerHTML = recommendations;

            // Show results section
            document.getElementById('results').style.display = 'block'; // Show results section
        } else {
            throw new Error('No valid data returned from server');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while analyzing the skin. Please try again.');
    });
});