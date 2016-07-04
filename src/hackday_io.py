import numpy as np
import sklearn.preprocessing
import os
import math
import random
from PIL import Image

src_folder = "../../GameOfThrones/"


def get_images_path():
    list_images_name = []
    with open(src_folder + "features/images.list", 'r') as f:
        for l in f.readlines():
            list_images_name.append(os.path.join(src_folder, 'images',  l[:-1]))
    return list_images_name


def load_image(image_list):
    for img in image_list:
        thumb = Image.open(img)
        thumb.thumbnail((36, 64), Image.ANTIALIAS)
        yield np.asarray(thumb)


def get_frames(list_scenes, num_eps=1):
    if num_eps != 1:
        raise NotImplementedError("The method get_frames has only been implemented for episode 1")
    list_scenes_name = ["scene_" + str(i) for i in list_scenes]
    list_images_name = []
    name_eps = []

    norm_features_path = src_folder + 'dump/features_frames_normalized.npy'
    norm_features = None
    if not os.path.exists(norm_features_path):
        feature_images = np.load(src_folder + "features/caffenet.npy")
        feature_audios = np.load(src_folder + "features/audio_1s.npy")
        print("Shape images: " + repr(feature_images.shape))
        print("Shape audios: " + repr(feature_audios.shape))
        concat_features = np.concatenate((feature_images, feature_audios), axis=1)
        norm_features = sklearn.preprocessing.normalize(concat_features)
        np.save(norm_features_path, norm_features)
    else:
        norm_features = np.load(norm_features_path)

    # je parcours la liste des path des images pour:
    # - la liste des paths des images
    # - la liste des noms des episodes
    with open(src_folder + "features/images.list", 'r') as f:
        for l in f.readlines():
            # la liste des paths de simages
            list_images_name.append(l[:-1])
            # la liste des noms des episodes
            name_eps.append(l.split('/')[0])

    # pour eviter le problème de 0 devant les chiffres (à part le 10)
    if num_eps >= 10:
        zero = ""
    else:
        zero = "0"
    list_frames = []
    # dans le fichier de scènes associé à l'episode cherché
    with open(src_folder + "scenes/" + "GameOfThrones.Season01.Episode" + zero + str(num_eps) + ".txt") as f2:
        # je regarde les timestamps de chaque scene
        for l in f2.readlines():
            # dict of episode: scenes: timestamps
            scene_name = l.split()[2]
            scene_timestamp_tuple = tuple([float(k) for k in l.split()[:2]])
            # si le numero de la scene est recherché
            frames = {"scene": None, "episode": num_eps, "vec": None}
            if scene_name in list_scenes_name:
                frames["scene"] = int(scene_name.split("_")[-1])

                frames_scene_float = scene_timestamp_tuple
                frames_scene_int = tuple([math.ceil(frames_scene_float[0]), math.ceil(frames_scene_float[1])])

                frames["vec"] = norm_features[frames_scene_int[0]:frames_scene_int[1]]
                list_frames.append(frames)

        return list_frames


def load_data(logger):
    list_images_name = []
    dict_eps_scenes_timestamps = {}
    name_eps = []

    with open(src_folder + "features/images.list", 'r') as f:
        for l in f.readlines():
            list_images_name.append(l[:-1])
            dict_eps_scenes_timestamps[l.split('/')[0]] = {}
            name_eps.append(l.split('/')[0])

    for eps in dict_eps_scenes_timestamps:
        with open(src_folder + "scenes/" + eps + ".txt") as f2:
            for l in f2.readlines():
                # dict of episode: scenes: timestamps
                dict_eps_scenes_timestamps[eps].update({l.split()[2]: tuple([float(k) for k in l.split()[:2]])})

    frames_scene_1_float = dict_eps_scenes_timestamps[name_eps[0]]["scene_1"]
    frames_scene_1_int = tuple([math.ceil(frames_scene_1_float[0]), int(frames_scene_1_float[1])])

    norm_features_path = src_folder + 'dump/features_frames_normalized.npy'

    if not os.path.exists(norm_features_path):
        feature_images = np.load(src_folder + "features/caffenet.npy")[frames_scene_1_int[0]:frames_scene_1_int[1]]
        feature_audios = np.load(src_folder + "features/audio_1s.npy")[frames_scene_1_int[0]:frames_scene_1_int[1]]
        logger.debug("Shape images: " + repr(feature_images.shape))
        logger.debug("Shape audios: " + repr(feature_audios.shape))

        concat_features = np.concatenate((feature_images, feature_audios), axis=1)
        norm_features = sklearn.preprocessing.normalize(concat_features)

        np.save(norm_features_path, norm_features)
    else:
        norm_features = np.load(norm_features_path)
    return norm_features, frames_scene_1_int


def generate_dataset(norm_features, size, frame_limits):

    list_tuples_x = []
    i = 0

    while i < size:

        frame_random_choice = random.randint(0, 1)

        if frame_random_choice:
            # si on choisi de prendre le frame x dans le cluster 0
            frame_random = norm_features[random.randint(0, frame_limits)]
            frame_random_plus = norm_features[random.randint(0, frame_limits)]
            frame_random_minus = norm_features[random.randint(frame_limits, len(norm_features) - 1)]
        else:
            # si on choisi de prendre le frame x dans le cluster 1
            frame_random = norm_features[random.randint(frame_limits, len(norm_features) - 1)]
            frame_random_plus = norm_features[random.randint(frame_limits, len(norm_features) - 1)]
            frame_random_minus = norm_features[random.randint(0, frame_limits)]

        # si jamais le x est identique au x+, on doit refaire (pour qu'ils soient differents)
        if (frame_random != frame_random_plus).any():
            i += 1
            list_tuples_x.append(tuple([frame_random, frame_random_plus, frame_random_minus]))
    return list_tuples_x

if __name__ == '__main__':
    for p in get_frames(range(3)):
        print(p)