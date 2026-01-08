/**
 * ai-facemaker Demo Application
 *
 * This handles the browser-based portrait generation using AWS Bedrock.
 */

// State
let cognitoUser = null;
let awsCredentials = null;

// DOM Elements
const authSection = document.getElementById('auth-section');
const generatorSection = document.getElementById('generator-section');
const resultsSection = document.getElementById('results-section');
const setupSection = document.getElementById('setup-section');

const signInForm = document.getElementById('sign-in-form');
const signedInView = document.getElementById('signed-in-view');
const authStatus = document.getElementById('auth-status');
const userEmailSpan = document.getElementById('user-email');

const generateBtn = document.getElementById('generate-btn');
const downloadBtn = document.getElementById('download-btn');
const regenerateBtn = document.getElementById('regenerate-btn');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error-message');
const resultImage = document.getElementById('result-image');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    if (!isConfigured()) {
        showStatus('Please configure AWS Cognito in aws-config.js', 'error');
        setupSection.style.display = 'block';
        return;
    }

    setupSection.style.display = 'none';
    initCognito();
    attachEventListeners();
});

// Initialize Cognito
function initCognito() {
    const poolData = {
        UserPoolId: AWS_CONFIG.userPoolId,
        ClientId: AWS_CONFIG.userPoolClientId,
    };

    const userPool = new AmazonCognitoIdentity.CognitoUserPool(poolData);
    cognitoUser = userPool.getCurrentUser();

    if (cognitoUser) {
        cognitoUser.getSession((err, session) => {
            if (err || !session.isValid()) {
                showSignInForm();
            } else {
                handleSignedIn(cognitoUser, session);
            }
        });
    } else {
        showSignInForm();
    }
}

// Event Listeners
function attachEventListeners() {
    document.getElementById('sign-in-btn').addEventListener('click', handleSignIn);
    document.getElementById('sign-up-btn').addEventListener('click', handleSignUp);
    document.getElementById('sign-out-btn').addEventListener('click', handleSignOut);
    generateBtn.addEventListener('click', handleGenerate);
    downloadBtn.addEventListener('click', handleDownload);
    regenerateBtn.addEventListener('click', () => {
        resultsSection.style.display = 'none';
        generatorSection.style.display = 'block';
    });

    // Enter key on password field
    document.getElementById('password').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSignIn();
    });
}

// Auth Functions
function showSignInForm() {
    signInForm.style.display = 'block';
    signedInView.style.display = 'none';
    generatorSection.style.display = 'none';
}

function handleSignIn() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    if (!email || !password) {
        showStatus('Please enter email and password', 'error');
        return;
    }

    const authData = {
        Username: email,
        Password: password,
    };

    const authDetails = new AmazonCognitoIdentity.AuthenticationDetails(authData);

    const poolData = {
        UserPoolId: AWS_CONFIG.userPoolId,
        ClientId: AWS_CONFIG.userPoolClientId,
    };

    const userPool = new AmazonCognitoIdentity.CognitoUserPool(poolData);

    const userData = {
        Username: email,
        Pool: userPool,
    };

    cognitoUser = new AmazonCognitoIdentity.CognitoUser(userData);

    showStatus('Signing in...', '');

    cognitoUser.authenticateUser(authDetails, {
        onSuccess: (session) => {
            handleSignedIn(cognitoUser, session);
        },
        onFailure: (err) => {
            showStatus(err.message || 'Sign in failed', 'error');
        },
        newPasswordRequired: () => {
            showStatus('New password required. Please use AWS Console to set password.', 'error');
        },
    });
}

function handleSignUp() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    if (!email || !password) {
        showStatus('Please enter email and password', 'error');
        return;
    }

    const poolData = {
        UserPoolId: AWS_CONFIG.userPoolId,
        ClientId: AWS_CONFIG.userPoolClientId,
    };

    const userPool = new AmazonCognitoIdentity.CognitoUserPool(poolData);

    const attributeList = [
        new AmazonCognitoIdentity.CognitoUserAttribute({
            Name: 'email',
            Value: email,
        }),
    ];

    showStatus('Creating account...', '');

    userPool.signUp(email, password, attributeList, null, (err, result) => {
        if (err) {
            showStatus(err.message || 'Sign up failed', 'error');
            return;
        }
        showStatus('Account created! Check your email for verification code.', 'success');
    });
}

function handleSignedIn(user, session) {
    const email = user.getUsername();
    userEmailSpan.textContent = email;

    signInForm.style.display = 'none';
    signedInView.style.display = 'block';
    generatorSection.style.display = 'block';
    showStatus('Signed in successfully', 'success');

    // Get AWS credentials
    getAWSCredentials(session);
}

function handleSignOut() {
    if (cognitoUser) {
        cognitoUser.signOut();
    }
    cognitoUser = null;
    awsCredentials = null;
    showSignInForm();
    showStatus('Signed out', '');
}

function getAWSCredentials(session) {
    const idToken = session.getIdToken().getJwtToken();

    AWS.config.region = AWS_CONFIG.region;

    const loginKey = `cognito-idp.${AWS_CONFIG.region}.amazonaws.com/${AWS_CONFIG.userPoolId}`;

    AWS.config.credentials = new AWS.CognitoIdentityCredentials({
        IdentityPoolId: AWS_CONFIG.identityPoolId,
        Logins: {
            [loginKey]: idToken,
        },
    });

    AWS.config.credentials.refresh((err) => {
        if (err) {
            showStatus('Failed to get AWS credentials: ' + err.message, 'error');
        } else {
            awsCredentials = AWS.config.credentials;
        }
    });
}

// Generation Functions
async function handleGenerate() {
    if (!awsCredentials) {
        showStatus('Not authenticated. Please sign in.', 'error');
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
    const bedrock = new AWS.BedrockRuntime({ region: AWS_CONFIG.region });

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

    // Convert base64 to bytes and resize if needed
    const imageBytes = Uint8Array.from(atob(base64Image), c => c.charCodeAt(0));

    // For browser, we'll display at native size and let CSS handle scaling
    // In production, you'd want server-side processing for proper downscaling
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
    authStatus.textContent = message;
    authStatus.className = 'status ' + type;
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}
