const API_BASE_URL = window.location.origin; // This will automatically use the correct URL whether local or deployed
let conversationId = null;

const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');

// Add message to chat
function addMessage(content, isUser = false, sources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    if (isUser) {
        messageContent.textContent = content;
    } else {
        // Parse bot response as Markdown
        messageContent.innerHTML = marked.parse(content);
    }
    
    messageDiv.appendChild(messageContent);
    
    if (!isUser && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sourcesDiv.textContent = `Sources: ${sources.join(', ')}`;
        messageDiv.appendChild(sourcesDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.id = 'loadingIndicator';
    
    const loadingDots = document.createElement('div');
    loadingDots.className = 'loading-dots';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'loading-dot';
        loadingDots.appendChild(dot);
    }
    
    loadingDiv.appendChild(loadingDots);
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideLoading() {
    const loadingIndicator = document.getElementById('loadingIndicator');
    if (loadingIndicator) {
        loadingIndicator.remove();
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = `Error: ${message}`;
    chatMessages.appendChild(errorDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage(message) {
    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, conversation_id: conversationId })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to send message');
        }

        const data = await response.json();
        conversationId = data.conversation_id;
        return { response: data.response, sources: data.sources || [] };
    } catch (error) {
        console.error('Error sending message:', error);
        throw error;
    }
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const message = messageInput.value.trim();
    if (!message) return;
    
    addMessage(message, true);
    messageInput.value = '';
    sendButton.disabled = true;
    messageInput.disabled = true;
    showLoading();
    
    try {
        const result = await sendMessage(message);
        hideLoading();
        addMessage(result.response, false, result.sources);
    } catch (error) {
        hideLoading();
        showError(error.message);
    } finally {
        sendButton.disabled = false;
        messageInput.disabled = false;
        messageInput.focus();
    }
});

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) showError('API is not responding properly');
    } catch {
        showError('Cannot connect to API. Make sure the server is running on localhost:8000');
    }
}

window.addEventListener('load', () => {
    messageInput.focus();
    checkHealth();
});

messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});
