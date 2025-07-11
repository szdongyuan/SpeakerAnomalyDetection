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
    def copy_with_selected_wave(audio_data_path_list: list, output_dir: str = None):
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
    def packaging_with_dir(target_dir_path, output_name=None, progress_callback=None):
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
                return error_code.INVALID_PATH, f"Directory does not exist: '{dir_path}'"
        except Exception as e:
            return error_code.INVALID_DELETE, f"Failed to delete directory: {str(e)[:40]}"
