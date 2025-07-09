import os
import shutil

from consts import error_code
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
