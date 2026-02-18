// Basic JavaScript for GreenClassify

// Example: Confirm before submitting
document.querySelector("form").addEventListener("submit", function (e) {
  const fileInput = document.getElementById("file");
  if (!fileInput.files.length) {
    alert("Please select an image file.");
    e.preventDefault();
  }
});
