function chooseFile() {
    document.getElementById("fileInput").click(); // Trigger file input click
}

function uploadFile() {
    let fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
        alert("Please select a file to upload.");
        return;
    }

    showLoader();
    
    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoader();
        if (data.success) {
            alert("File uploaded successfully!");
            displayResults(data.results);
            
            // Split graphs into sections of 2 per row
            let graphHtml = "";
            for (let i = 0; i < data.graphs.length; i += 2) {
                graphHtml += `<div class="graph-row">`;
                graphHtml += `<img src="/get_graph/${data.graphs[i]}" class="graph-img">`;
                if (data.graphs[i + 1]) {
                    graphHtml += `<img src="/get_graph/${data.graphs[i + 1]}" class="graph-img">`;
                }
                graphHtml += `</div>`;
            }
            document.getElementById("graphs").innerHTML = graphHtml;

            alert("Analysis complete");
            showChatButton();
        } else {
            document.getElementById("output").innerText = "Error: " + data.error;
        }
    })
    .catch(error => {
        hideLoader();
        console.error("Error:", error);
        alert("An error occurred while uploading the file. Please try again.");
    });
}

function displayResults(data) {
    let outputDiv = document.getElementById("output");
    
    let html = `<h2> </h2>`;
    html += `<h2>Model Accuracy</h2>`
    html += `<p><strong>ANN Accuracy:</strong> ${data["ANN Accuracy"].toFixed(4)}</p>`;
    html += `<p><strong>Hybrid Model Accuracy:</strong> ${data["Hybrid Model Accuracy"].toFixed(4)}</p>`;
    html += `<p><strong>XGBoost Accuracy:</strong> ${data["XGBoost Accuracy"].toFixed(4)}</p>`;
    html += `<br></br>`;
    html += `<h2>Classification Report</h2>`;
    html += `<table border="1" cellspacing="0" cellpadding="5">
                <tr>
                    <th>Class</th>
                    <th>F1-Score</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Support</th>
                </tr>`;
    for (let key in data["Classification Report"]) {
        if (!["accuracy", "macro avg", "weighted avg"].includes(key)) {
            let report = data["Classification Report"][key];
            html += `<tr>
                        <td>${key}</td>
                        <td>${report["f1-score"].toFixed(4)}</td>
                        <td>${report["precision"].toFixed(4)}</td>
                        <td>${report["recall"].toFixed(4)}</td>
                        <td>${report["support"]}</td>
                    </tr>`;
        }
    }
    html += `</table>`;
    html += `<br></br>`;
    html += `<h2>Confusion Matrix</h2>`;
    html += `<table border="1" cellspacing="0" cellpadding="5">`;
    data["Confusion Matrix"].forEach(row => {
        html += `<tr><td>${row.join("</td><td>")}</td></tr>`;
    });
    html += `</table>`;

    outputDiv.innerHTML = html;
}

function runModel() {
    showLoader(); // Show loader when the model starts running
    uploadFile(); // Trigger file upload and analysis
}

function showLoader() {
    document.getElementById("loader-overlay").style.display = "flex";
    document.body.classList.add("blurred"); // Apply blur effect
}

function hideLoader() {
    document.getElementById("loader-overlay").style.display = "none";
    document.body.classList.remove("blurred"); // Remove blur effect
}

function showChatBox() {
    const chatBox = document.getElementById("chatBox");
    chatBox.style.display = "flex"; // Show chat box
    document.getElementById("chatInput").focus(); // Focus on the input field
}

function closeChatBox() {
    const chatBox = document.getElementById("chatBox");
    chatBox.style.display = "none"; // Hide chat box
    showChatButton(); // Ensure chat button is visible again
}

function showChatButton() {
    const chatButton = document.getElementById("chatButton");
    chatButton.style.display = "flex"; // Show chat button
    chatButton.innerHTML = "🤖 Chat with AI"; // Button text
    chatButton.onclick = showChatBox; // Bind click event to show chat
}


function sendMessage() {
    const userMessage = document.getElementById("chatInput").value;
    if (!userMessage.trim()) {
        alert("Please enter a message.");
        return;
    }
    fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userMessage })
    })
    .then(response => response.json())
    .then(data => {
        const chatOutput = document.getElementById("chatOutput");
        chatOutput.innerHTML += `<div class="user-message">You: ${userMessage}</div>`;
        chatOutput.innerHTML += `<div class="ai-message">AI: ${data.reply}</div>`;
        document.getElementById("chatInput").value = ""; // Clear input field
        chatOutput.scrollTop = chatOutput.scrollHeight; // Auto-scroll to bottom
    })
    .catch(error => {
        console.error("Error:", error);
        alert("An error occurred while sending the message.");
    });
}
