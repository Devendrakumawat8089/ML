import random

from app.core.config import (
    CHALLENGER_TRAFFIC_PERCENT
)


def choose_model():

    random_number = random.random()

    if random_number < CHALLENGER_TRAFFIC_PERCENT:

        return "challenger"

    return "champion"