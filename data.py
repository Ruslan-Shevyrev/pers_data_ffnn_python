import oracledb
import numpy as np
import utils
import config.config_db as config_db
import config.config as config
import sql

input_data = []
output_data = []


def get_study_data():
    input_data.clear()
    output_data.clear()
    i = 1
    with oracledb.connect(user=config_db.USER,
                          password=config_db.PASSWORD,
                          dsn=config_db.DSN) as connection:
        with connection.cursor() as cursor:
            for r in cursor.execute(sql.get_study_select()):
                list_num_out = np.zeros(config.OUTPUT_SIZE, np.float32)

                list_num_out[r[1]] = 1
                input_data.append(utils.str_to_arr(r[0]))

                output_data.append(list_num_out)
                if i % 1000 == 0:
                    print(i)
                i += 1
        return input_data, output_data


def get_predict_data():
    input_data.clear()
    with oracledb.connect(user=config_db.USER,
                          password=config_db.PASSWORD,
                          dsn=config_db.DSN) as connection:
        with connection.cursor() as cursor:

            for r in cursor.execute(sql.get_predict_select()):
                input_data.append(r[0])
    return input_data


if __name__ == '__main__':
    input_data, output_data = get_study_data()
    print(input_data, output_data)
