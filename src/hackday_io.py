import numpy as np
import sklearn.preprocessing
import os
import math
import random
from PIL import Image

src_folder = "../../GameOfThrones/"


def get_images_path():
    global src_folder
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
    """
    Return the list of dict for each requested episodes and scenes.

    This method is only available for episode 1 at the moment.

    :param list_scenes: a list of integers [int]
    :param num_eps: an integer
    :return: The list of dicts: [{"episode": int, "scene": int, "vec": np.array}]
    """
    global src_folder
    # FOr the moment this method only works for num_eps == 1
    if num_eps != 1:
        raise NotImplementedError("The method get_frames has only been implemented for episode 1")

    # the feature.npy is no more dumped/loaded
    feature_images = np.load(src_folder + "features/caffenet.npy")
    feature_audios = np.load(src_folder + "features/audio_1s.npy")
    print("Shape images: " + repr(feature_images.shape))
    print("Shape audios: " + repr(feature_audios.shape))
    concat_features = np.concatenate((feature_images, feature_audios), axis=1)
    number_of_features = concat_features[0].size

    # je parcours la liste des path des images pour récupérer la liste des paths des images
    list_images_name = []
    with open(src_folder + "features/images.list", 'r') as f:
        for l in f.readlines():
            # la liste des paths de simages
            list_images_name.append(l[:-1])

    # pour eviter le problème de 0 devant les chiffres (en dessous de 10)
    if num_eps >= 10:
        zero = ""
    else:
        zero = "0"
    list_frames_locations = []
    matrix_interesting_frames = np.empty((0, number_of_features))
    int_matrix_frames_pointer = 0
    # dans le fichier de scènes associé à l'episode cherché
    with open(src_folder + "scenes/" + "GameOfThrones.Season01.Episode" + zero + str(num_eps) + ".txt") as f2:
        # je regarde les timestamps de chaque scene
        for l in f2.readlines():
            scene_name = l.split()[2]
            try:
                scene_number = int(scene_name.split("_")[-1])
            except ValueError:
                print("The scene " + scene_name + " doesn't follow the pattern: '^scene_[0-9]+$'")
                continue
            # si le numero de la scene est recherché
            if scene_number in list_scenes:
                scene_timestamp_tuple = tuple([float(k) for k in l.split()[:2]])
                # the scene bounds timestamp are given with floating point, we need integer bounds
                frames_scene_float = scene_timestamp_tuple
                # the tuple contain the time (int) of the first frame after the begining of the scene and the
                # last frame before the end of the scene. This way, all the frames of the scene are taken but the last
                # frame overlaps the begining of the next frame
                frames_scene_int = tuple([math.ceil(frames_scene_float[0]), int(frames_scene_float[1])])
                # the dict of frames is built with the indexes of the related frames in the matrix of interesting frames
                # the int stored here are counted from 1 and not from 0, beware when taking them in the features matrix
                index_first_frame = frames_scene_int[0] - 1
                index_last_frame = frames_scene_int[1] - 1
                # explicit is better than implicit: i want the last frame to be included
                number_of_taken_frames = index_last_frame + 1 - index_first_frame
                interesting_frames = concat_features[index_first_frame: index_last_frame + 1]
                matrix_interesting_frames = np.append(matrix_interesting_frames, interesting_frames, axis=0)

                # given this dict, we will take back the normalized frame in the matrix with the formula: matrix[first: last]
                dict_frames_locations = dict(scene=scene_number,
                                             episode=num_eps,
                                             first=int_matrix_frames_pointer,
                                             last=int_matrix_frames_pointer + number_of_taken_frames)
                list_frames_locations.append(dict_frames_locations)

    # after all the interesting frames have been isolated, we have to normalize them
    norm_features = sklearn.preprocessing.normalize(matrix_interesting_frames)
    # the list containing all the requested dict: {"scene": int, "episode": int, "vec": np.array} which will be returned
    list_frames = []
    for frame in list_frames_locations:
        first = frame["first"]
        last = frame["last"]
        dict_frame = dict(scene=frame["scene"],
                          episode=frame["episode"],
                          vec=norm_features[first: last])
        print("Episode: " + str(dict_frame["episode"]) + "; Scene: " + str(dict_frame["scene"]) + "; Vec Shape: " + str(dict_frame["vec"].shape))
        list_frames.append(dict_frame)
    return list_frames


def load_data(logger):
    """
    Obsolete
    """
    global src_folder
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
    for p in get_frames(range(1, 2)):
        print(p)