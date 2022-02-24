import os

import src.data.convert as convert
import src.data.prepare as pr
import src.models.model_classes as cl

root_dir = os.path.abspath("..")

if __name__ == '__main__':
    convert.to_csv()
    pr.download_model()
    cl.default_config()
