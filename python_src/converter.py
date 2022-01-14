import json
import os

from bounding_box import BoundingBox
from enumerators import BBFormat, BBType, CoordinatesType

def get_files_dir(directory, extensions=['*']):
    ret = []
    for extension in extensions:
        if extension == '*':
            ret += [f for f in os.listdir(directory)]
            continue
        elif extension is None:
            # accepts all extensions
            extension = ''
        elif '.' not in extension:
            extension = f'.{extension}'
        ret += [f for f in os.listdir(directory) if f.lower().endswith(extension.lower())]
    return ret
    
def get_files_recursively(directory, extension="*"):
    files = [
        os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(directory)
        for f in get_files_dir(directory, [extension])
    ]
    # Disconsider hidden files, such as .DS_Store in the MAC OS
    ret = [f for f in files if not os.path.basename(f).startswith('.')]
    return ret

def get_file_name_only(file_path):
    if file_path is None:
        return ''
    return os.path.splitext(os.path.basename(file_path))[0]

def _get_annotation_files(file_path):
    # Path can be a directory containing all files or a directory containing multiple files
    if file_path is None:
        return []
    annotation_files = []
    if os.path.isfile(file_path):
        annotation_files = [file_path]
    elif os.path.isdir(file_path):
        annotation_files = get_files_recursively(file_path)
    return sorted(annotation_files)

def is_json(file_path):
    """ Verify by the extension if a given file path represents a json file.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file ends with .json, False otherwise.
    """
    return os.path.splitext(file_path)[-1].lower() == '.json'

def get_all_keys(items):
    """ Get all keys in a list of dictionary.

    Parameters
    ----------
    items : list
        List of dictionaries.

    Returns
    -------
    list
        List containing all keys in the dictionary.
    """
    ret = []
    if not hasattr(items, '__iter__'):
        return ret
    if isinstance(items, str):
        return ret
    for i, item in enumerate(items):
        if isinstance(item, list):
            ret.append(get_all_keys(item))
        elif isinstance(item, dict):
            [ret.append(it) for it in item.keys() if it not in ret]
    return ret

def json_contains_tags(file_path, tags):
    """ Verify if a given JSON file contains all tags in a list.

    Parameters
    ----------
    file_path : str
        Path of the file.
    tags : list
        List containing strings representing the tags to be found.

    Returns
    -------
    bool
        True if XML file contains all tags, False otherwise.
    """
    with open(file_path, "r") as f:
        json_object = json.load(f)
    all_keys = []
    for key, item in json_object.items():
        keys = get_all_keys(item)
        if len(keys) == 0:
            all_keys.append(key)
        for k in keys:
            all_keys.append(f'{key}/{k}')

    tags_matching = 0
    for tag in tags:
        if tag in all_keys:
            tags_matching += 1
    return tags_matching == len(tags)

def is_coco_format(file_path):
    """ Verify if a given file path represents a file with annotations in coco format.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file contains annotations in coco format, False otherwise.
    """
    return is_json(file_path) and json_contains_tags(file_path, [
        'annotations/bbox',
        'annotations/image_id',
    ])

def coco2bb(path, bb_type=BBType.GROUND_TRUTH):
    ret = []
    # Get annotation files in the path
    annotation_files = _get_annotation_files(path)
    # Loop through each file
    for file_path in annotation_files:
        if not is_coco_format(file_path):
            continue

        with open(file_path, "r") as f:
            json_object = json.load(f)

        # COCO json file contains basically 3 lists:
        # categories: containing the classes
        # images: containing information of the images (width, height and filename)
        # annotations: containing information of the bounding boxes (x1, y1, bb_width, bb_height)
        classes = {}
        if 'categories' in json_object:
            classes = json_object['categories']
            # into dictionary
            classes = {c['id']: c['name'] for c in classes}
        images = {}
        # into dictionary
        for i in json_object['images']:
            images[i['id']] = {
                'file_name': i['file_name'],
                'img_size': (int(i['width']), int(i['height']))
            }
        annotations = []
        if 'annotations' in json_object:
            annotations = json_object['annotations']

        for annotation in annotations:
            img_id = annotation['image_id']
            x1, y1, bb_width, bb_height = annotation['bbox']
            if bb_type == BBType.DETECTED and 'score' not in annotation.keys():
                print('Warning: Confidence not found in the JSON file!')
                return ret
            confidence = annotation['score'] if bb_type == BBType.DETECTED else None
            # Make image name only the filename, without extension
            img_name = images[img_id]['file_name']
            img_name = get_file_name_only(img_name)
            # create BoundingBox object
            bb = BoundingBox(image_name=img_name,
                             class_id=classes[annotation['category_id']],
                             coordinates=(x1, y1, bb_width, bb_height),
                             type_coordinates=CoordinatesType.ABSOLUTE,
                             img_size=images[img_id]['img_size'],
                             confidence=confidence,
                             bb_type=bb_type,
                             format=BBFormat.XYWH)
            ret.append(bb)
    return ret