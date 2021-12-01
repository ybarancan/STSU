import os


def _get_env_var(varname):
    value = os.getenv(varname)
    if not value:
        raise EnvironmentError("Required environment variable '{}' is not set.".format(varname))
    return value


class CocoPaths(object):
    def __init__(self):
        raise ValueError("Static class 'CocoPaths' should not be instantiated")

    @staticmethod
    def images_dir():
        return _get_env_var('COCO_TRAIN_IMAGES_DIR')

    @staticmethod
    def ids_file():
        return os.path.join(_get_env_var('STEMSEG_JSON_ANNOTATIONS_DIR'), 'coco.json')


class YoutubeVISPaths(object):
    def __init__(self):
        raise ValueError("Static class 'YoutubeVISPaths' should not be instantiated")

    @staticmethod
    def training_base_dir():
        return os.path.join(_get_env_var('YOUTUBE_VIS_BASE_DIR'), 'train')

    @staticmethod
    def val_base_dir():
        return os.path.join(_get_env_var('YOUTUBE_VIS_BASE_DIR'), 'valid')

    @staticmethod
    def train_vds_file():
        return os.path.join(_get_env_var('STEMSEG_JSON_ANNOTATIONS_DIR'), 'youtube_vis_train.json')

    @staticmethod
    def val_vds_file():
        return os.path.join(_get_env_var('STEMSEG_JSON_ANNOTATIONS_DIR'), 'youtube_vis_val.json')


class DavisUnsupervisedPaths(object):
    def __init__(self):
        raise ValueError("Static class 'DavisUnsupervisedPaths' should not be instantiated")

    @staticmethod
    def trainval_base_dir():
        return _get_env_var('DAVIS_BASE_DIR')

    @staticmethod
    def train_vds_file():
        return os.path.join(_get_env_var('STEMSEG_JSON_ANNOTATIONS_DIR'), 'davis_train.json')

    @staticmethod
    def val_vds_file():
        return os.path.join(_get_env_var('STEMSEG_JSON_ANNOTATIONS_DIR'), 'davis_val.json')


class KITTIMOTSPaths(object):
    def __init__(self):
        raise ValueError("Static class 'KITTIMOTSPaths' should not be instantiated")

    @staticmethod
    def train_images_dir():
        return _get_env_var('KITTIMOTS_BASE_DIR')

    @staticmethod
    def train_vds_file():
        return os.path.join(_get_env_var('STEMSEG_JSON_ANNOTATIONS_DIR'), 'kittimots_train.json')

    @staticmethod
    def val_vds_file():
        return os.path.join(_get_env_var('STEMSEG_JSON_ANNOTATIONS_DIR'), 'kittimots_val.json')


class MapillaryPaths(object):
    def __init__(self):
        raise ValueError("Static class 'MapillaryPaths' should not be instantiated")

    @staticmethod
    def images_dir():
        return _get_env_var('MAPILLARY_IMAGES_DIR')

    @staticmethod
    def ids_file():
        return os.path.join(_get_env_var('STEMSEG_JSON_ANNOTATIONS_DIR'), 'mapillary.json')


class PascalVOCPaths(object):
    def __init__(self):
        raise ValueError("Static class 'PascalVOCPaths' should not be instantiated")

    @staticmethod
    def images_dir():
        return os.path.join(_get_env_var('PASCAL_VOC_IMAGES_DIR'))

    @staticmethod
    def ids_file():
        return os.path.join(_get_env_var('STEMSEG_JSON_ANNOTATIONS_DIR'), 'pascal_voc.json')
