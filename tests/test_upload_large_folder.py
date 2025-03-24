import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from huggingface_hub import HfApi


class TestUploadLargeFolder(unittest.TestCase):
    def setUp(self):
        self.api = HfApi()
        self.repo_id = "test-repo"
        self.folder_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.folder_path)

    def test_upload_large_folder_with_symlinks(self):
        # Create a folder structure with symlinks
        os.makedirs(os.path.join(self.folder_path, "subfolder"))
        with open(os.path.join(self.folder_path, "file1.txt"), "w") as f:
            f.write("content1")
        os.symlink(
            os.path.join(self.folder_path, "file1.txt"),
            os.path.join(self.folder_path, "subfolder", "symlink1.txt"),
        )

        # Mock the API calls
        with patch.object(self.api, "create_repo") as mock_create_repo, patch.object(
            self.api, "create_commit"
        ) as mock_create_commit:
            mock_create_repo.return_value.repo_id = self.repo_id

            # Call the upload_large_folder function with recurse_symlinks=True
            self.api.upload_large_folder(
                repo_id=self.repo_id,
                folder_path=self.folder_path,
                repo_type="model",
                recurse_symlinks=True,
            )

            # Check that the symlinked file was uploaded
            uploaded_files = [
                call[1]["operations"][0].path_in_repo
                for call in mock_create_commit.call_args_list
            ]
            self.assertIn("subfolder/symlink1.txt", uploaded_files)

    def test_upload_large_folder_without_symlinks(self):
        # Create a folder structure with symlinks
        os.makedirs(os.path.join(self.folder_path, "subfolder"))
        with open(os.path.join(self.folder_path, "file1.txt"), "w") as f:
            f.write("content1")
        os.symlink(
            os.path.join(self.folder_path, "file1.txt"),
            os.path.join(self.folder_path, "subfolder", "symlink1.txt"),
        )

        # Mock the API calls
        with patch.object(self.api, "create_repo") as mock_create_repo, patch.object(
            self.api, "create_commit"
        ) as mock_create_commit:
            mock_create_repo.return_value.repo_id = self.repo_id

            # Call the upload_large_folder function with recurse_symlinks=False
            self.api.upload_large_folder(
                repo_id=self.repo_id,
                folder_path=self.folder_path,
                repo_type="model",
                recurse_symlinks=False,
            )

            # Check that the symlinked file was not uploaded
            uploaded_files = [
                call[1]["operations"][0].path_in_repo
                for call in mock_create_commit.call_args_list
            ]
            self.assertNotIn("subfolder/symlink1.txt", uploaded_files)
