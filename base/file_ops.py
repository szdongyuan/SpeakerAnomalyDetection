import os
import shutil

from consts import error_code


class FileOps(object):
    """
        A static method class for file operations.
    """
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
            shutil.rmtree(dest_dir) # shutil.rmtree() 递归地删除文件
            os.mkdir(dest_dir)
            os.mkdir(dest_dir + "/OK")
            os.mkdir(dest_dir + "/NG")
            return error_code.OK, "finish creating empty okng dir"
        except Exception as e:
            err_msg = "failed to create [%s], %s" % (dest_dir, str(e)[:40])
            return error_code.INVALID_PATH, err_msg
