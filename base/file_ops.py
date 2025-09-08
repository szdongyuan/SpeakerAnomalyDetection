import os
import shutil
import time
from zipfile import ZipFile, ZIP_DEFLATED

from consts import error_code, model_consts
from consts.running_consts import DEFAULT_DIR


class FileOps(object):

    @staticmethod
    def create_empty_okng(dest_dir):
        """
        Create an empty directory with 'OK' and 'NG' subdirectories.

        Args:
        - dest_dir: string
            The destination directory path of the file to be copied.

        Returns:
        - error_code.OK: int
            The code indicating a successful operation.
        - error_code.INVALID_PATH: int
            The code indicating a failure operation.
        - err_msg: string
            The error message.
        """
        try:
            shutil.rmtree(dest_dir)  # shutil.rmtree() remove files
            os.mkdir(dest_dir)
            os.mkdir(dest_dir + "/OK")
            os.mkdir(dest_dir + "/NG")
            return error_code.OK, "finish creating empty okng dir"
        except Exception as e:
            err_msg = "failed to create [%s], %s" % (dest_dir, str(e)[:40])
            return error_code.INVALID_PATH, err_msg

    @staticmethod
    def get_relative_path(file_path: str, base_path: str = DEFAULT_DIR):
        """
        Get the relative path of a file with respect to a base path.
        """
        relative_path = os.path.relpath(file_path, base_path)
        relative_path = relative_path.replace("\\", "/")
        relative_path = relative_path.replace("../", "")
        return relative_path

    @staticmethod
    def ensure_directory_exists(save_path: str):
        """
        Ensure that the directory where the save path resides exists.
        Args:
            save_path: str
                The save path of audio signals.
        Returns:
        """
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def copy_with_selected_wav(audio_data_path_list: list, output_dir: str = None):
        """
        Copy selected audio files to the specified output directory.

        If no output directory is provided, a default folder named with
        'package_YYYYMMDD' will be created under the package storage path.

        Files are copied into subfolders based on their label:
        - 'NG'  -> copied to ./NG/
        - 'OK'  -> copied to ./OK/
        - 'not_labeled' -> copied to ./not_labeled/
        - 'other' -> copied to ./

        Parameters:
            audio_data_path_list (list): List of file paths to be copied.
            output_dir (str, optional): Target directory for copying files.
                                        Defaults to a new 'package_YYYYMMDD' folder.

        Returns:
            str: Path of the output directory where files were copied.
        """

        if output_dir is None:
            output_dir = os.path.join(model_consts.STORED_PACKAGE_PATH, "package_" + time.strftime("%Y%m%d"))
        count = 1
        temp_dir = output_dir

        while os.path.exists(output_dir):
            output_dir = temp_dir + f"({str(count)})"
            count += 1

        for audio_data_path in audio_data_path_list:
            if "NG" in audio_data_path:
                target_dir_path = os.path.join(output_dir, "NG")
            elif "OK" in audio_data_path:
                target_dir_path = os.path.join(output_dir, "OK")
            elif "not_labeled" in audio_data_path:
                target_dir_path = os.path.join(output_dir, "not_labeled")
            else:
                target_dir_path = output_dir
            audio_data_path = DEFAULT_DIR + audio_data_path
            os.makedirs(target_dir_path, exist_ok=True)
            if "db" in audio_data_path:
                dirname, filename = os.path.split(audio_data_path)
                target_dir_path = target_dir_path.replace("\\", "/")
                dir_name = target_dir_path.split("/")[-1]
                filename = filename.split(".")[0]
                new_file_name = target_dir_path + "/" + filename + "_" + dir_name + ".db"
                shutil.copy2(audio_data_path, new_file_name)
            else:
                shutil.copy2(audio_data_path, target_dir_path)
        return output_dir

    @staticmethod
    def archive_with_dir(target_dir_path, output_name=None, progress_callback=None):
        """
        This method compresses the contents of the specified directory into an archive
        file using the given output name and format.
        """
        if output_name is None:
            target_dir_path = target_dir_path.replace("\\", "/")
            output_name = target_dir_path.split("/")[-1]
        count = 1
        output_name = model_consts.STORED_PACKAGE_PATH + "/" + output_name
        temp_name = output_name
        while os.path.exists(output_name + ".zip"):
            output_name = temp_name + f"({str(count)})"
            count += 1
        output_zip = output_name + ".zip"

        files = []
        for root, dirs, fs in os.walk(target_dir_path):
            for f in fs:
                files.append(os.path.join(root, f))

        total_files_num = len(files)
        progressed = 0

        with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zip_file:
            for file in files:
                # arcname = os.path.relpath(file, os.path.dirname(target_dir_path))
                arcname = os.path.relpath(file, target_dir_path)
                zip_file.write(file, arcname)
                progressed += 1
                if progress_callback:
                    progress_callback(progressed, total_files_num)

        return output_zip

    @staticmethod
    def create_zip_with_files(file_path_list, output_zip_path=None, base_dir=DEFAULT_DIR, categorize=True, progress_callback=None):
        """
        Create a zip archive that contains the given files. zip archive will contains OK/NG/not_labeled subfolders originally

        Parameters
        ----------
        file_path_list : list[str]
            Paths of files to be included in the archive. Relative paths will be
            evaluated against ``base_dir``.
        output_zip_path : str, optional
            Full path of the output zip file. When omitted a file named
            ``export_YYYYMMDD.zip`` is created under
            ``model_consts.STORED_PACKAGE_PATH``.
        base_dir : str, optional
            Base directory for resolving relative paths. Defaults to ``DEFAULT_DIR``.
        categorize : bool, optional
            If True, files whose path contains 'NG', 'OK' or 'not_labeled' will be
            placed into corresponding sub-folders in the archive. Set False to keep
            all files in the root of the archive.
        progress_callback : callable, optional
            A callback that receives two integers (progress, total) so that UI
            elements such as progress bars can be updated while the archive is
            being created.

        Returns
        -------
        str
            Absolute path to the created zip file.
        """
        if output_zip_path is None:
            output_zip_path = os.path.join(
                model_consts.STORED_PACKAGE_PATH, f"export_{time.strftime('%Y%m%d')}.zip"
            )

        if not output_zip_path.endswith(".zip"):
            output_zip_path += ".zip"

        os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)

        total_files = len(file_path_list)
        progressed = 0

        category_rules = {"NG": "NG", "OK": "OK", "not_labeled": "not_labeled"} if categorize else {}

        with ZipFile(output_zip_path, "w", compression=ZIP_DEFLATED) as zip_file:
            for path in file_path_list:
                full_path = path if os.path.isabs(path) else os.path.join(base_dir, path)
                rel_path = os.path.relpath(full_path, base_dir)

                arcname = os.path.basename(rel_path)
                for key, folder in category_rules.items():
                    if key in rel_path:
                        arcname = os.path.join(folder, os.path.basename(rel_path))
                        break

                if rel_path.endswith(".db"):
                    arcname = os.path.basename(rel_path)

                if os.path.exists(full_path):
                    zip_file.write(full_path, arcname)
                else:
                    pass

                progressed += 1
                if progress_callback:
                    progress_callback(progressed, total_files)

        return output_zip_path

    @staticmethod
    def delete_directory(dir_path):
        """
        Recursively deletes a directory and all its contents.

        This method attempts to remove the specified directory and all files
        and subdirectories within it. If the directory does not exist, an
        error code is returned. Any exceptions during deletion are caught and
        returned with a descriptive message.

        Parameters:
            dir_path (str): The full path of the directory to be deleted.

        Returns:
            tuple: A tuple containing:
                - int: An error code indicating the result.
                - str: A message describing the outcome or error.
        """
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                return error_code.OK, f"Directory '{dir_path}' has been successfully deleted."
            else:
                return error_code.OK, f"Directory does not exist: '{dir_path}'"
        except Exception as e:
            return error_code.INVALID_DELETE, f"Failed to delete directory: {str(e)[:40]}"
            
    @staticmethod
    def move_wav_to_dir(recorded_path, label):
        """
        Move the recorded WAV file to the directory corresponding to its label.

        This function moves the recorded audio file to a predefined directory structure based on its label (OK/NG).
        If the target directories do not exist, they will be created automatically.

        Args:
            recorded_path (str): The full path of the recorded file
            label (str): File label, should be either "OK" or "NG"

        Returns:
            str: The full path of the file after moving, or an empty string if the filename is empty
        """
        dir_paths = [
            model_consts.STORED_RECORDED_UNLABELED_PATH,
            model_consts.STORED_RECORDED_OK_PATH,
            model_consts.STORED_RECORDED_NG_PATH,
        ]
        for path in dir_paths:
            if not os.path.exists(path):
                os.makedirs(path)
        file_name = os.path.basename(recorded_path)
        target_path = ""
        if file_name:
            if label == "OK":
                target_path = model_consts.STORED_RECORDED_OK_PATH + "/" + file_name
            elif label == "NG":
                target_path = model_consts.STORED_RECORDED_NG_PATH + "/" + file_name
            else:
                return
            shutil.move(recorded_path, target_path)
        return target_path
