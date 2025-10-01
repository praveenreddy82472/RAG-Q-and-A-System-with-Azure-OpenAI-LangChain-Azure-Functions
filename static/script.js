const sendBtn = document.getElementById("send-btn");
const userInput = document.getElementById("user-input");
const chatWindow = document.getElementById("chat-window");

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", e => { if (e.key === "Enter") sendMessage(); });

async function sendMessage() {
    const question = userInput.value.trim();
    if (!question) return;
    appendMessage(question, "user");
    userInput.value = "";
    appendMessage("Bot is typing...", "bot", true);

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });
        const data = await response.json();
        removeTyping();
        appendMessage(data.answer, "bot");
    } catch (err) {
        removeTyping();
        appendMessage("Error fetching response", "bot");
    }
}

function appendMessage(text, sender, isTyping=false) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${sender}-msg`;
    msgDiv.innerText = text;
    if(isTyping) msgDiv.id = "typing-msg";
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeTyping() {
    const typingMsg = document.getElementById("typing-msg");
    if (typingMsg) typingMsg.remove();
}
