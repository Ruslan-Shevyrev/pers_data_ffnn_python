import config.config as config

STUDY_SELECT = ("SELECT GOOD_VAL, "
                "       CHECK_ID,"
                "       GOOD_ID "
                "   FROM MV_GOOD_DATA_FOR_EDUCATION ")

PREDICT_SELECT = ("SELECT VAL "
                  " FROM PREDICT_TABLE ")


def get_study_select():
    if config.PARTITION_DATA:
        return STUDY_SELECT + \
            "   WHERE ROWNUM_RANDOM >= "+str(config.STUDY_FROM_ID) + \
            "       AND ROWNUM_RANDOM < "+str(config.STUDY_TO_ID)
    else:
        return STUDY_SELECT


def get_predict_select():
    return PREDICT_SELECT
