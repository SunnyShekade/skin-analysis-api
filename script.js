// Handle login form submission
document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission

    // Get user input
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value.trim();

    // Here you can add your logic to handle sign-in
    // For now, we'll just log the input and redirect to the skin analysis page
    console.log('Username:', username);
    console.log('Password:', password);

    // Simple validation (you can replace this with actual authentication logic)
    if (username && password) {
        // Redirect to the skin analysis page
        window.location.href = 'skin_analysis.html'; // Redirect to the skin analysis page
    } else {
        alert('Please enter both username and password.');
    }
});

// Handle signup form toggle
document.getElementById('showSignup').addEventListener('click', function() {
    document.querySelector('.form-container').style.display = 'none';
    document.getElementById('signupContainer').style.display = 'block';
});

document.getElementById('showLogin').addEventListener('click', function() {
    document.getElementById('signupContainer').style.display = 'none';
    document.querySelector('.form-container').style.display = 'block';
});