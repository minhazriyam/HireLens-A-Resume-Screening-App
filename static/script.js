const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("resumes");
const fileList = document.getElementById("fileList");
const browseBtn = dropArea.querySelector(".browse");

// Click -> open file dialog
browseBtn.addEventListener("click", () => fileInput.click());

// Drag & drop effects
["dragenter", "dragover"].forEach((event) => {
  dropArea.addEventListener(event, (e) => {
    e.preventDefault();
    dropArea.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((event) => {
  dropArea.addEventListener(event, (e) => {
    e.preventDefault();
    dropArea.classList.remove("dragover");
  });
});

// Drop handling
dropArea.addEventListener("drop", (e) => {
  fileInput.files = e.dataTransfer.files;
  displayFiles(fileInput.files);
});

// File input change
fileInput.addEventListener("change", () => displayFiles(fileInput.files));

function displayFiles(files) {
  fileList.innerHTML = "";
  Array.from(files).forEach((file) => {
    const li = document.createElement("li");
    li.textContent = file.name;
    fileList.appendChild(li);
  });
}
