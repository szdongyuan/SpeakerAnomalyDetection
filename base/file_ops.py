import os
import shutil

from consts import error_code


class FileOps(object):

    @staticmethod
    def create_empty_okng(dest_dir):
        try:
            shutil.rmtree(dest_dir)
            os.mkdir(dest_dir)
            os.mkdir(dest_dir + "/OK")
            os.mkdir(dest_dir + "/NG")
            return error_code.OK, "finish creating empty okng dir"
        except Exception as e:
            err_msg = "failed to create [%s], %s" % (dest_dir, str(e)[:40])
            return error_code.INVALID_PATH,
