import FFNN as pdl

#pdl.training_new_model(output_model='FFNN_pers_data_100000')

#pdl.training_exists_model(input_model='FFNN_pers_data_1900000', output_model='FFNN_pers_data_2200000')

#pdl.test_predict(model_name='FFNN_pers_data_2200000.h5')

test = pdl.test_predict('FFNN_pers_data_2200000.h5')
print(test)

#pdl.convert_to_h5(input_model='FFNN_pers_data_2200000')
