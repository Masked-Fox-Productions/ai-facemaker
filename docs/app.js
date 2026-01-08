/**
 * ai-facemaker Demo Application
 *
 * Browser-based portrait generation using AWS Bedrock with user-provided credentials.
 * Credentials are stored in browser memory only and never sent to any server.
 */

// State
let awsCredentials = null;
let awsRegion = null;

// DOM Elements
const credentialsSection = document.getElementById('credentials-section');
const generatorSection = document.getElementById('generator-section');
const resultsSection = document.getElementById('results-section');
const setupSection = document.getElementById('setup-section');

const credentialsForm = document.getElementById('credentials-form');
const connectedView = document.getElementById('connected-view');
const credentialsStatus = document.getElementById('credentials-status');
const connectedRegion = document.getElementById('connected-region');

const generateBtn = document.getElementById('generate-btn');
const downloadBtn = document.getElementById('download-btn');
const regenerateBtn = document.getElementById('regenerate-btn');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error-message');
const resultImage = document.getElementById('result-image');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    attachEventListeners();
});

// Event Listeners
function attachEventListeners() {
    document.getElementById('connect-btn').addEventListener('click', handleConnect);
    document.getElementById('disconnect-btn').addEventListener('click', handleDisconnect);
    generateBtn.addEventListener('click', handleGenerate);
    downloadBtn.addEventListener('click', handleDownload);
    regenerateBtn.addEventListener('click', () => {
        resultsSection.style.display = 'none';
        generatorSection.style.display = 'block';
    });

    // Toggle visibility buttons
    document.querySelectorAll('.toggle-visibility').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-target');
            const input = document.getElementById(targetId);
            if (input.type === 'password') {
                input.type = 'text';
                btn.textContent = '🔒';
                btn.title = 'Hide';
            } else {
                input.type = 'password';
                btn.textContent = '👁';
                btn.title = 'Show';
            }
        });
    });

    // Enter key on secret key field
    document.getElementById('secret-access-key').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleConnect();
    });
}

// Connection Functions
async function handleConnect() {
    const region = document.getElementById('aws-region').value;
    const accessKeyId = document.getElementById('access-key-id').value.trim();
    const secretAccessKey = document.getElementById('secret-access-key').value.trim();

    if (!accessKeyId || !secretAccessKey) {
        showStatus('Please enter both Access Key ID and Secret Access Key', 'error');
        return;
    }

    if (!accessKeyId.startsWith('AKIA') && !accessKeyId.startsWith('ASIA')) {
        showStatus('Access Key ID should start with AKIA or ASIA', 'error');
        return;
    }

    showStatus('Validating credentials...', '');

    // Configure AWS SDK
    AWS.config.update({
        region: region,
        credentials: new AWS.Credentials(accessKeyId, secretAccessKey)
    });

    // Test the credentials by listing foundation models
    const bedrock = new AWS.Bedrock({ region: region });

    try {
        await bedrock.listFoundationModels({ byOutputModality: 'IMAGE' }).promise();

        // Credentials work
        awsCredentials = { accessKeyId, secretAccessKey };
        awsRegion = region;

        showConnectedState();
        showStatus('Connected successfully', 'success');
    } catch (err) {
        showStatus('Connection failed: ' + (err.message || 'Invalid credentials'), 'error');
        awsCredentials = null;
        awsRegion = null;
    }
}

function handleDisconnect() {
    awsCredentials = null;
    awsRegion = null;
    AWS.config.credentials = null;

    // Clear form fields
    document.getElementById('access-key-id').value = '';
    document.getElementById('secret-access-key').value = '';

    showDisconnectedState();
    showStatus('Disconnected', '');
}

function showConnectedState() {
    credentialsForm.style.display = 'none';
    connectedView.style.display = 'block';
    connectedRegion.textContent = `(${awsRegion})`;
    generatorSection.style.display = 'block';
}

function showDisconnectedState() {
    credentialsForm.style.display = 'block';
    connectedView.style.display = 'none';
    generatorSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

// Generation Functions
async function handleGenerate() {
    if (!awsCredentials) {
        showStatus('Not connected. Please enter your AWS credentials.', 'error');
        return;
    }

    // Get form values
    const worldContext = document.getElementById('world-context').value;
    const worldStyle = document.getElementById('world-style').value;
    const negative = document.getElementById('negative-prompt').value;
    const charName = document.getElementById('char-name').value;
    const charRole = document.getElementById('char-role').value;
    const charDesc = document.getElementById('char-description').value;
    const model = document.getElementById('model-select').value;
    const size = parseInt(document.getElementById('size-select').value);
    const frame = document.getElementById('prompt-frame').value;

    // Validate
    if (!worldContext || !worldStyle || !charName) {
        showError('Please fill in world context, style, and character name');
        return;
    }

    // Build prompt
    const prompt = buildPrompt(worldContext, worldStyle, charName, charRole, charDesc, frame);

    // Show loading
    generateBtn.disabled = true;
    loadingDiv.style.display = 'block';
    errorDiv.style.display = 'none';

    try {
        const imageBytes = await generateImage(model, prompt, negative, size);
        displayResult(imageBytes);
    } catch (err) {
        showError(err.message || 'Generation failed');
    } finally {
        generateBtn.disabled = false;
        loadingDiv.style.display = 'none';
    }
}

function buildPrompt(context, style, name, role, description, frame) {
    return `${frame}

Character: ${name}, ${role}.
${description}

Setting: ${context.split('.').slice(0, 2).join('.')}.

Style: ${style}`;
}

async function generateImage(modelId, prompt, negative, size) {
    const bedrock = new AWS.BedrockRuntime({ region: awsRegion });

    let requestBody;

    if (modelId.startsWith('amazon.titan')) {
        // Titan request format
        requestBody = {
            taskType: 'TEXT_IMAGE',
            textToImageParams: {
                text: prompt.substring(0, 512),
            },
            imageGenerationConfig: {
                numberOfImages: 1,
                quality: 'premium',
                height: 1024,
                width: 1024,
                cfgScale: 8.0,
                seed: Math.floor(Math.random() * 2147483646),
            },
        };
        if (negative) {
            requestBody.textToImageParams.negativeText = negative.substring(0, 512);
        }
    } else if (modelId.startsWith('stability.sd3')) {
        // SD3.5 request format
        requestBody = {
            prompt: prompt.substring(0, 10000),
            mode: 'text-to-image',
            aspect_ratio: '1:1',
            output_format: 'png',
            seed: Math.floor(Math.random() * 4294967294),
        };
        if (negative) {
            requestBody.negative_prompt = negative;
        }
    } else {
        // SDXL request format
        requestBody = {
            text_prompts: [
                { text: prompt.substring(0, 2000), weight: 1.0 },
            ],
            cfg_scale: 7,
            seed: Math.floor(Math.random() * 4294967294),
            steps: 50,
            width: 1024,
            height: 1024,
        };
        if (negative) {
            requestBody.text_prompts.push({ text: negative, weight: -1.0 });
        }
    }

    const response = await bedrock.invokeModel({
        modelId: modelId,
        body: JSON.stringify(requestBody),
        accept: 'application/json',
        contentType: 'application/json',
    }).promise();

    const responseBody = JSON.parse(new TextDecoder().decode(response.body));

    // Extract image based on model
    let base64Image;
    if (responseBody.images) {
        base64Image = responseBody.images[0];
    } else if (responseBody.artifacts) {
        base64Image = responseBody.artifacts[0].base64;
    } else {
        throw new Error('No image in response');
    }

    // Convert base64 to bytes
    const imageBytes = Uint8Array.from(atob(base64Image), c => c.charCodeAt(0));

    return imageBytes;
}

function displayResult(imageBytes) {
    const blob = new Blob([imageBytes], { type: 'image/png' });
    const url = URL.createObjectURL(blob);
    resultImage.src = url;
    resultImage.dataset.blob = url;

    generatorSection.style.display = 'none';
    resultsSection.style.display = 'block';
}

function handleDownload() {
    const url = resultImage.dataset.blob;
    if (!url) return;

    const charName = document.getElementById('char-name').value || 'portrait';
    const safeName = charName.toLowerCase().replace(/\s+/g, '_');

    const a = document.createElement('a');
    a.href = url;
    a.download = `${safeName}_portrait.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Utility Functions
function showStatus(message, type) {
    credentialsStatus.textContent = message;
    credentialsStatus.className = 'status ' + type;
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}
