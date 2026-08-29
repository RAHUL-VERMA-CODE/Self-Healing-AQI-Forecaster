import sys


class customException(Exception):

    def __init__(self, error_message, error_details: sys):
        super().__init__(str(error_message))

        self.error_message = error_message

        _, _, exc_tb = error_details.exc_info()

        if exc_tb is not None:
            self.lineo = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineo = None
            self.file_name = None

    def __str__(self):
        return (
            "Error occurred in python script "
            "name [{0}] line number [{1}] "
            "error message [{2}]".format(
                self.file_name,
                self.lineo,
                str(self.error_message)
            )
        )