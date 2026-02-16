# Branch Protection Setup Guide

This guide walks you through securing the `main` branch on GitHub.

## Why Branch Protection?

Branch protection prevents:
- Direct pushes to main (forces PRs for review)
- Merging code that fails tests
- Accidental deletion of the main branch
- Force pushes that rewrite history

## Step-by-Step Instructions

### 1. Navigate to Branch Protection Settings

1. Go to your repository on GitHub: `https://github.com/AlbanOtt/chemometrics_vibe_coding_public`
2. Click **Settings** (top navigation)
3. In the left sidebar, click **Branches**
4. Under "Branch protection rules", click **Add rule** (or **Add branch protection rule**)

### 2. Configure Protection Rules

#### Basic Settings

**Branch name pattern:** `main`

#### Required Settings (Recommended)

✅ **Require a pull request before merging**
- Check: "Require approvals" (set to 1 or more)
- Check: "Dismiss stale pull request approvals when new commits are pushed"
- Check: "Require review from Code Owners" (optional, requires CODEOWNERS file)

✅ **Require status checks to pass before merging**
- Check: "Require branches to be up to date before merging"
- Search and select these status checks:
  - `test (3.9)` - Python 3.9 tests
  - `test (3.10)` - Python 3.10 tests
  - `test (3.11)` - Python 3.11 tests
  - `test (3.12)` - Python 3.12 tests
  - `lint` - Code quality checks
  - `quality-gate` - Overall quality gate
  - `build` - Book build check (from deploy-book.yml)

  **Note:** Status checks will only appear in the list after they've run at least once. Push your changes first, then return to add them.

✅ **Require conversation resolution before merging**
- Ensures all PR comments are addressed

✅ **Require signed commits** (optional but recommended)
- Ensures commits are cryptographically verified

✅ **Require linear history** (optional)
- Prevents merge commits, requires rebase or squash

✅ **Do not allow bypassing the above settings**
- Applies rules even to administrators

#### Additional Protections

✅ **Lock branch** (not recommended for active development)
- Makes branch read-only

✅ **Allow force pushes** - Leave **UNCHECKED**
- Prevents force pushes that rewrite history

✅ **Allow deletions** - Leave **UNCHECKED**
- Prevents accidental branch deletion

### 3. Save Protection Rules

Click **Create** (or **Save changes**)

## For Repository Administrators

If you need to make emergency fixes, you can:

1. **Temporarily disable branch protection:**
   - Settings → Branches → Edit rule → Uncheck rules → Save
   - Make your changes
   - Re-enable protection

2. **Use the bypass option** (if you didn't check "Do not allow bypassing"):
   - Administrators can push directly even with protection enabled
   - **Not recommended** except for emergencies

## Recommended Workflow After Protection

### Normal Development

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes, commit
git add .
git commit -m "feat: add my feature"

# Push to remote
git push -u origin feature/my-feature

# Create PR on GitHub
# Wait for CI checks to pass
# Request review
# Merge via GitHub UI
```

### For Solo Development

Even when working alone, branch protection is valuable:

```bash
# Create PR branch
git checkout -b update/fix-typo

# Make and commit changes
git commit -m "docs: fix typo in README"

# Push and create PR
git push -u origin update/fix-typo

# On GitHub: Create PR, wait for checks, merge
```

This ensures:
- All code is tested before reaching main
- Git history is clean and auditable
- CI catches issues before deployment

## Verification

After setup, verify protection is working:

1. Try to push directly to main (should fail):
   ```bash
   git checkout main
   git commit --allow-empty -m "test"
   git push
   # Should see: "remote: error: GH006: Protected branch update failed"
   ```

2. Create a test PR and verify:
   - Status checks run automatically
   - Merge button is blocked until checks pass
   - Approval is required (if configured)

## CODEOWNERS (Optional Enhancement)

Create `.github/CODEOWNERS` to automatically request reviews:

```
# Default owners for everything
* @AlbanOtt

# Python code requires review
*.py @AlbanOtt

# Documentation can be edited by docs team
*.md @AlbanOtt
*.qmd @AlbanOtt

# CI/CD changes require special attention
.github/workflows/* @AlbanOtt
```

## Troubleshooting

**Status checks don't appear in the list:**
- They only appear after running at least once
- Push this branch first to trigger workflows
- Wait for workflows to complete
- Then return to branch protection settings

**Can't merge even though checks pass:**
- Ensure branch is up to date with main
- Check that all required status checks are green
- Verify all conversations are resolved

**Emergency hotfix needed:**
- Create a hotfix branch anyway
- Create PR marked as "hotfix"
- Expedite review process
- Don't bypass protection rules

## Next Steps

After setting up branch protection:

1. [ ] Test the protection by creating a test PR
2. [ ] Update team documentation about the new workflow
3. [ ] Configure GitHub notifications for PR reviews
4. [ ] Consider setting up CODEOWNERS
5. [ ] Document any exceptions or special cases
