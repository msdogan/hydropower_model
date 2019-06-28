__author__ = 'mustafa_dogan'
from model import *

plants = [
		'Shasta',
		'Keswick',
		# 'Oroville',
		# 'Bullards Bar',
		# 'Englebright',
		'Folsom',
		'Nimbus',
		'New Melones',
		# 'Don Pedro',
		# 'New Exchequer',
		'Pine Flat'
		]


# simulation start time: year, month, day, hour, minute
start = [2010,10,1,0,0]
# simulation end time: year, month, day, hour, minute
end = [2017,10,1,0,0]

# resampling frequency for creating averages. A: annual, M: monthly, W: weekly, D: daily, H: hourly
step = ['M']

# energy price data
price_path = 'inputs/price.csv'
# inflow data
flow_path = 'inputs/inflow/inflow_cms.csv'

# you must run the model in defined time-step (frequency) first
# before using warmstart option  in order to get initial values of variables
warmstart = True

# call hydropower model class
hp_model = HYDROPOWER(start,end,step,overall_average=False,warmstart=warmstart,flow_path=flow_path,price_path=price_path,plants=plants)

# ************ Linear Model ************

# preprocess to create input network matrix
hp_model.preprocess_LP(plot_benefit_curves=False)

# create pyomo hydropower lp model
hp_model.create_pyomo_LP(datadir='model/data_lp.csv',display_model=False)

# solve lp model
hp_model.solve_pyomo_LP(solver='glpk',stream_solver=True,display_model_out=False,display_raw_results=False)

# postprocess and save results as csv time-series
hp_model.postprocess(save_path='outputs/linear_model')

# ************ Nonlinear Model ************

# preprocess to create input network matrix
hp_model.preprocess_NLP(warmstart_path='outputs/nonlinear_model')

# # print network schematic
# hp_model.print_schematic(datadir='model/data_nlp.csv',ts_condensed=True)

# create pyomo hydropower model
hp_model.create_pyomo_NLP(datadir='model/data_nlp.csv',display_model=False)

# solve the model
hp_model.solve_pyomo_NLP(solver='ipopt',stream_solver=True,display_model_out=False,display_raw_results=False,max_iter=3000,mu_init=1e-9) 

# postprocess and save results as csv time-series
hp_model.postprocess(save_path='outputs/nonlinear_model')
