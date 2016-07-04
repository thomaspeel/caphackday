import logging

from hackday_io import generate_dataset, load_data
from tripletnetwork import learn

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.DEBUG)
    steam_handler.setFormatter(formatter)
    logger.addHandler(steam_handler)

    norm_features, frames_scene_1_int = load_data(logger)

    logger.debug("Shape images-audios concatenation: " + repr(norm_features.shape))

    #dernière frame de la scène du tunnel
    frame_limits = 59 - frames_scene_1_int[0]

    list_tuples_x = generate_dataset(norm_features, size=10000, frame_limits=frame_limits)

    learn(list_tuples_x, batchsize=256, num_epochs=16)
