/**
 * AWS Cognito Configuration for ai-facemaker Demo
 *
 * To use this demo, you need to set up AWS Cognito:
 *
 * 1. Create a Cognito User Pool:
 *    - Go to AWS Console > Cognito > User Pools > Create user pool
 *    - Enable email sign-in
 *    - Create an app client (no secret)
 *    - Note the User Pool ID and App Client ID
 *
 * 2. Create a Cognito Identity Pool:
 *    - Go to AWS Console > Cognito > Identity Pools > Create identity pool
 *    - Enable authenticated access
 *    - Connect to your User Pool
 *    - Note the Identity Pool ID
 *
 * 3. Configure IAM Role:
 *    - The authenticated role should have this policy:
 *    {
 *      "Version": "2012-10-17",
 *      "Statement": [{
 *        "Effect": "Allow",
 *        "Action": "bedrock:InvokeModel",
 *        "Resource": "arn:aws:bedrock:*::foundation-model/*"
 *      }]
 *    }
 *
 * 4. Update the values below with your configuration
 */

const AWS_CONFIG = {
    // AWS Region where your Cognito pools are created
    region: 'us-east-1',

    // Cognito User Pool ID (format: us-east-1_XXXXXXXXX)
    userPoolId: 'YOUR_USER_POOL_ID',

    // Cognito User Pool App Client ID
    userPoolClientId: 'YOUR_APP_CLIENT_ID',

    // Cognito Identity Pool ID (format: us-east-1:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    identityPoolId: 'YOUR_IDENTITY_POOL_ID',

    // Default Bedrock model to use
    defaultModel: 'amazon.titan-image-generator-v1',
};

// Check if configured
function isConfigured() {
    return (
        AWS_CONFIG.userPoolId !== 'YOUR_USER_POOL_ID' &&
        AWS_CONFIG.userPoolClientId !== 'YOUR_APP_CLIENT_ID' &&
        AWS_CONFIG.identityPoolId !== 'YOUR_IDENTITY_POOL_ID'
    );
}
