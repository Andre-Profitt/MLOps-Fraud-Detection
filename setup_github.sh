#!/bin/bash

# 🚀 GitHub Repository Setup Script
# This script helps you push your MLOps Fraud Detection project to GitHub

set -e  # Exit on error

echo "============================================"
echo "🚀 GitHub Repository Setup"
echo "============================================"
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Step 1: Get GitHub username
echo "📝 Step 1: GitHub Account Information"
echo "--------------------------------------"
read -p "Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ Error: GitHub username is required"
    exit 1
fi

# Step 2: Get user information for git
echo ""
echo "📝 Step 2: Git Configuration"
echo "--------------------------------------"
read -p "Enter your name (for git commits): " GIT_NAME
read -p "Enter your email (for git commits): " GIT_EMAIL

if [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ]; then
    echo "❌ Error: Name and email are required"
    exit 1
fi

# Step 3: Configure git
echo ""
echo "⚙️  Configuring git..."
git config user.name "$GIT_NAME"
git config user.email "$GIT_EMAIL"
echo "✅ Git configured"

# Step 4: Check if remote already exists
echo ""
echo "🔍 Checking existing remotes..."
if git remote | grep -q "origin"; then
    echo "⚠️  Remote 'origin' already exists. Removing it..."
    git remote remove origin
fi

# Step 5: Add new remote
REPO_URL="https://github.com/${GITHUB_USERNAME}/MLOps-Fraud-Detection.git"
echo ""
echo "🔗 Adding remote repository..."
echo "Repository URL: $REPO_URL"
git remote add origin "$REPO_URL"
echo "✅ Remote added"

# Step 6: Verify commit exists
echo ""
echo "🔍 Verifying commit..."
if ! git log -1 > /dev/null 2>&1; then
    echo "❌ No commits found. Creating initial commit..."
    git add -A
    git commit -m "Initial commit: Complete MLOps Fraud Detection Pipeline"
fi
echo "✅ Commit verified"

# Step 7: Push to GitHub
echo ""
echo "============================================"
echo "📤 Ready to push to GitHub!"
echo "============================================"
echo ""
echo "Repository: $REPO_URL"
echo ""
echo "⚠️  IMPORTANT: Before continuing, make sure you have:"
echo "   1. Created the repository 'MLOps-Fraud-Detection' on GitHub"
echo "   2. NOT initialized it with README, .gitignore, or license"
echo ""
read -p "Have you created the repository on GitHub? (y/n): " CREATED

if [ "$CREATED" != "y" ] && [ "$CREATED" != "Y" ]; then
    echo ""
    echo "Please create the repository first:"
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: MLOps-Fraud-Detection"
    echo "3. Make it Public or Private"
    echo "4. DO NOT initialize with README, .gitignore, or license"
    echo "5. Click 'Create repository'"
    echo ""
    echo "Then run this script again!"
    exit 0
fi

# Push to GitHub
echo ""
echo "📤 Pushing to GitHub..."
if git push -u origin main; then
    echo ""
    echo "============================================"
    echo "🎉 SUCCESS!"
    echo "============================================"
    echo ""
    echo "✅ Your project is now on GitHub!"
    echo "🔗 View it at: https://github.com/${GITHUB_USERNAME}/MLOps-Fraud-Detection"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Visit your repository"
    echo "   2. Add topics: mlops, mlflow, fraud-detection, fastapi"
    echo "   3. Star your own repo ⭐"
    echo "   4. Share on LinkedIn!"
    echo ""
else
    echo ""
    echo "============================================"
    echo "❌ Push failed"
    echo "============================================"
    echo ""
    echo "Possible reasons:"
    echo "1. Repository doesn't exist on GitHub"
    echo "2. Repository was initialized with files"
    echo "3. Authentication failed"
    echo ""
    echo "Try:"
    echo "1. Make sure the repository exists: https://github.com/${GITHUB_USERNAME}/MLOps-Fraud-Detection"
    echo "2. If it has files, use: git push -u origin main --force"
    echo "3. Check your GitHub credentials"
    echo ""
fi
