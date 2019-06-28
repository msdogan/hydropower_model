from __future__ import division
__author__ = 'mustafa_dogan'
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np 
import pandas as pd
import os, csv, warnings, re, datetime
from pyomo.environ import * # requires pyomo installation
from pyomo.opt import SolverFactory
from pyomo.core import Constraint
warnings.simplefilter('ignore')

'''
hydropower equation: Power=e*rho*g*Q*H,
where e is efficiency, rho is density of water, 
g is gravitational constant, Q is discharge and H is head

for SI units
Power (Watt) = e * rho (kg/m**3) * g (m/s**2) * Q (m**3/s) * H (m)
Power (MW) = e * 1000 * 9.81 * Q * H / 10**6

for English unit after conversion
Power (MW) = [e * Q (cfs) * H (ft)] / [1.181 * 10**4]

Generation (MWh) = Power (MW) * hour (h)
Revenue ($) = Generation (MWh) * Energy Price ($/MWh)
'''

class HYDROPOWER():

    # initialize global parameters
    def __init__(self,start,end,step,overall_average=True,warmstart=False,flow_path='inputs/inflow/inflow_cms.csv',price_path='inputs/price.csv',plants=[
                                    # 'Trinity',
                                    # 'Carr',
                                    # 'Spring Creek',
                                    'Shasta',
                                    'Keswick',
                                    'Oroville',
                                    'Thermalito',
                                    'Bullards Bar',
                                    'Englebright',
                                    'Folsom',
                                    'Nimbus',
                                    'New Melones',
                                    'Don Pedro',
                                    'New Exchequer',
                                    'Pine Flat'
                                    ]):
        
        print('initializing the model V1.0')

        self.warmstart = warmstart

        self.CDEC_station_locator={
                            'Shasta':'SHA',
                            'Keswick':'KES',
                            'Oroville':'ORO',
                            'Bullards Bar':'BUL',
                            # 'Englebright':'ENG',
                            'Folsom':'FOL',
                            'Nimbus':'NAT',
                            'New Melones':'NML',
                            # 'Don Pedro':'DNP',
                            # 'New Exchequer':'EXC',
                            'Pine Flat':'PNF'                          
                            }

        # time-step information to be used for converting m3/s to m3 for mass balance
        self.freq = step[-1]
        self.total_seconds = {'H':3600,'D':86400,'W':604800,'M':2592000,'A':31556952}
        self.conv_fac = self.total_seconds[self.freq]/1000000 # m3/s * conv_fac = million m3

        # network parameters: storage capacity, release capacity, efficiency, etc.
        network_param = pd.read_csv('inputs/network_properties.csv',header=0,index_col=0)
        self.network_param = network_param[plants]
        self.plants = plants

        st = datetime.datetime(start[0],start[1],start[2],start[3],start[4])
        en = datetime.datetime(end[0],end[1],end[2],end[3],end[4])

        # energy prices
        price = pd.read_csv(price_path,header=0,index_col=0)
        price.index = pd.to_datetime(price.index)
        price = price.sort_index(ascending=True)
        price = price['price ($/MWh)'].ix[st:en]
        price = price[~((price.index.month == 2) & (price.index.day == 29))] # remove February 29
        
        # inflow
        # flows to retrieve
        cdec_inf_plant = []
        [cdec_inf_plant.append(self.CDEC_station_locator[plant]) for plant in plants]
        flow = pd.read_csv(flow_path,header=0,index_col=0)
        flow.index = pd.to_datetime(flow.index)
        flow = pd.DataFrame(flow,columns=cdec_inf_plant).ix[st:en]
        flow = flow[~((flow.index.month == 2) & (flow.index.day == 29))] # remove February 29

        fl_index = {'H':flow.index.hour,'D':flow.index.day,'W':flow.index.week,'M':flow.index.month,'A':flow.index.year}
        prc_index = {'H':price.index.hour,'D':price.index.day,'W':price.index.week,'M':price.index.month,'A':price.index.year}
        
        # create averages for flow and price for defined frequencies
        if overall_average:
            # overall average based on frequencies
            flow = flow.groupby([fl_index[x] for x in step]).mean()
            price = price.groupby([prc_index[x] for x in step]).mean()
        else:
            # average based on a time-step frequency
            flow = flow.resample(self.freq).mean()
            flow = flow[~((flow.index.month == 2) & (flow.index.day == 29))] # remove February 29
            price = price.resample(self.freq).mean()
            price = price[~((price.index.month == 2) & (price.index.day == 29))] # remove February 29

        # save flow and price after time-step averaging
        flow.to_csv('outputs/average_flow.csv',header=True)
        price.to_csv('outputs/average_price.csv',header=True)


        # make sure lengths are matching
        if len(flow.index) != len(price.index):
          print('flow and price do not have the same index length but will continue to solve! '+str(len(flow.index))+' ,'+str(len(price.index)))

        self.price = price
        self.flow = flow

        # Constant Parameters
        self.rho = 1000 # density of water
        self.g = 9.81 # gravitational constant

        # hydropower revenue function
        global hydropower_rev
        def hydropower_rev(
                            convert, # m3/s to million m3, 1/convert million m3 to m3/s
                            f, # flow
                            s, # storage
                            e, # efficiency
                            rho, # density of water
                            g, # gravitational constant
                            a3,a2,a1,c, # polynomial parameters to calculate head from storage
                            h, # number of hours
                            p # energy price
                            ):
            return e*rho*g*f/convert*(a3*s**3+a2*s**2+a1*s+c)*h*p/1000000

        # network connectivities: [upstream node (i),downstream node (j)]
        self.network_conn = [
                    # ['Trinity','Carr'],
                    # ['Carr','Spring Creek'],
                    # ['Spring Creek','Keswick'],
                    ['Shasta','Keswick'],
                    ['Keswick','Delta'],
                    # ['Oroville','Thermalito'],
                    ['Oroville','Delta'],
                    # ['Thermalito','Delta'],
                    # ['Bullards Bar','Englebright'],
                    # ['Englebright','Delta'],
                    ['Bullards Bar','Delta'],
                    ['Folsom','Nimbus'],
                    ['Nimbus','Delta'],
                    ['New Melones','Delta'],
                    # ['Don Pedro','Delta'],
                    # ['New Exchequer','Delta'],
                    ['Pine Flat','Delta']
                    ]

        # a large number to represent infinity (million m3)
        self.inf_bound = 10**10
        # penalty (cost) for spilling
        self.spill_cost = -10**3 # ($)

    def preprocess_NLP(self,datadir='model/data_nlp.csv',warmstart_path='outputs/nonlinear_model'):
        print('*******\ncreating NLP network \n*******')
   
        network_param = self.network_param
        price = self.price
        flow = self.flow
        network_conn = self.network_conn
        inf_bound = self.inf_bound
        spill_cost = self.spill_cost

        # create timestamps from indices
        index = list(flow.index)
        index.append('ENDING')

        # default parameter values
        def_a3 = 0
        def_a2 = 0
        def_a1 = 0
        def_c = 0
        def_efficiency = 1
        def_price = 0
        def_a = 1
        def_lower_b = 0
        def_upper_b = inf_bound

        # an empty list to save network data        
        df_list = []
        # create links for intial storages
        df_list.append(['SUPERSOURCE','INITIAL',def_a3,def_a2,def_a1,def_c,def_efficiency,def_price,def_a,def_lower_b,def_upper_b])
        # create links for ending storages
        df_list.append(['ENDING','SUPERSINK',def_a3,def_a2,def_a1,def_c,def_efficiency,def_price,def_a,def_lower_b,def_upper_b])
        # write initial storage values
        for j in range(len(network_param.keys())):
            df_list.append(['INITIAL','stor_'+network_param.keys()[j]+'.'+str(index[0]),def_a3,def_a2,def_a1,def_c,def_efficiency,def_price,def_a,network_param[network_param.keys()[j]].loc['initial_storage (million m3)'],network_param[network_param.keys()[j]].loc['initial_storage (million m3)']])
        for i in range(len(index)-1):
            # create subsource links for each time-step from supersource
            df_list.append(['SUPERSOURCE','INFLOW.'+str(index[i]),def_a3,def_a2,def_a1,def_c,def_efficiency,def_price,def_a,def_lower_b,def_upper_b])
            # create subsink links for each time-step from supersink
            df_list.append(['DELTA.'+str(index[i]),'SUPERSINK',def_a3,def_a2,def_a1,def_c,def_efficiency,def_price,def_a,def_lower_b,def_upper_b])
            for j in range(len(network_param.keys())):
                # from supersource, create inflows to plants (constrained lb=ub)
                df_list.append(['INFLOW.'+str(index[i]),'stor_'+network_param.keys()[j]+'.'+str(index[i]),def_a3,def_a2,def_a1,def_c,def_efficiency,def_price,def_a,flow[self.CDEC_station_locator[network_param.keys()[j]]].iloc[i]*self.conv_fac,flow[self.CDEC_station_locator[network_param.keys()[j]]].iloc[i]*self.conv_fac])
                # write storage properties (ub=0 if no storage capacity)
                df_list.append(['stor_'+network_param.keys()[j]+'.'+str(index[i]),
                                str(index[i+1]) if str(index[i+1]) == 'ENDING' else 'stor_'+network_param.keys()[j]+'.'+str(index[i+1]),
                                # polynomial function to calculate head (m) from storage (million m3)
                                # head = a3*stor^3+a2*stor^2+a1*stor+c
                                network_param[network_param.keys()[j]].loc['a3'],
                                network_param[network_param.keys()[j]].loc['a2'],
                                network_param[network_param.keys()[j]].loc['a1'],
                                network_param[network_param.keys()[j]].loc['c'],
                                network_param[network_param.keys()[j]].loc['efficiency'],
                                def_price,
                                def_a if self.freq == 'H' else network_param[network_param.keys()[j]].loc['evap_coeff'], # use default amplitude if hourly time-step
                                network_param[network_param.keys()[j]].loc['ending_storage (million m3)'] if str(index[i+1]) == 'ENDING' else network_param[network_param.keys()[j]].loc['deadpool (million m3)'],
                                network_param[network_param.keys()[j]].loc['ending_storage (million m3)'] if str(index[i+1]) == 'ENDING' else network_param[network_param.keys()[j]].loc['storage_capacity (million m3)']])
                # write turbine release links (energy prices are here)
                df_list.append(['stor_'+network_param.keys()[j]+'.'+str(index[i]),'flow_'+network_param.keys()[j]+'.'+str(index[i]),def_a3,def_a2,def_a1,def_c,def_efficiency,price.iloc[i],def_a,def_lower_b,network_param[network_param.keys()[j]].loc['release_capacity (cms)']*self.conv_fac])
                # write spill links (penalties for spilling)
                df_list.append(['stor_'+network_param.keys()[j]+'.'+str(index[i]),'spill_'+network_param.keys()[j]+'.'+str(index[i]),def_a3,def_a2,def_a1,def_c,def_efficiency,spill_cost,def_a,def_lower_b,def_upper_b])
                # combine spills and turbine releases downstream
                df_list.append(['spill_'+network_param.keys()[j]+'.'+str(index[i]),'flow_'+network_param.keys()[j]+'.'+str(index[i]),def_a3,def_a2,def_a1,def_c,def_efficiency,def_price,def_a,def_lower_b,def_upper_b])
                # connect reservoir downstream flow nodes to sink or downstream reservoir
                connected = False
                i1 = network_param.keys()[j]
                for i2 in network_param.columns:
                    if [i1,i2] in network_conn:
                        connected = True
                        df_list.append(['flow_'+i1+'.'+str(index[i]),'stor_'+i2+'.'+str(index[i]),def_a3,def_a2,def_a1,def_c,def_efficiency,def_price,def_a,def_lower_b,def_upper_b])
                        break
                if not connected:
                    df_list.append(['flow_'+i1+'.'+str(index[i]),'DELTA.'+str(index[i]),def_a3,def_a2,def_a1,def_c,def_efficiency,def_price,def_a,def_lower_b,def_upper_b])
        
        datafile = pd.DataFrame(df_list,columns=['i','j','a3','a2','a1','c','efficiency','price','amplitude','lower_bound','upper_bound'])
        
        if self.warmstart:
            # get decision variables from previous solution
            fl = pd.read_csv(warmstart_path+'/unsorted_f.csv',header=0)
            st = pd.read_csv(warmstart_path+'/unsorted_s.csv',header=0)
            link_init = pd.concat([fl, st],ignore_index=True)
            
            ws_values = [] 
            for link in df_list:
                got_value=False
                for index in link_init.index:
                    if link[0] in link_init['link'][index].split(', ')[0] and link[1] in link_init['link'][index].split(', ')[1]:
                        ws_values.append(link_init['value'][index])
                        link_init.drop(index,inplace=True)
                        got_value=True
                        break
                if not got_value:
                    ws_values.append(0)
            datafile['warmstart'] = ws_values
        datafile.to_csv(datadir,index=False)
        print('nonlinear model network data has been exported')

    def print_schematic(self,datadir='model/data_nlp.csv',ts_condensed=False):
        # *************
        # Plot simplified and detailed network schematic and plant locations on a map
        # *************

        from mpl_toolkits.basemap import Basemap # requires Basemap installation
        plt.style.use('bmh')
        from graphviz import Digraph # requires graphviz installation
        print('printing network schematics')

        # plot condensed network
        g = Digraph('G',format='pdf',filename='schematic/network_schematic.gv')
        g.attr(size='6,6',label='Simplified Network Schematic',fontsize='12')
        g.node_attr.update(color='lightblue2', style='filled',shape='ellipse')
        for n in self.network_conn:
            g.edge(n[0], n[1])
        g.view()

        # whole detailed network
        detailed_network=pd.read_csv(datadir)
        g = Digraph('G',format='pdf',filename='schematic/detailed_network.gv')
        g.attr(label='Detailed Network Schematic',fontsize='20')
        for l in detailed_network.index:
            if ts_condensed: # remove time step info and represent as another link (for simplicity)
                g.edge(detailed_network['i'][l].split('.')[0], detailed_network['j'][l].split('.')[0])               
            else: # print all nodes and links
                g.edge(detailed_network['i'][l], detailed_network['j'][l])
        g.view()

        # plot facilities locations on a map
        fig = plt.figure(figsize=(5.5,5))
        ax = plt.gca()
        # change resolution to i: intermadiate for faster plotting. (i: intermediate, h: high)
        m = Basemap(projection='merc',llcrnrlon=-125,llcrnrlat=32,urcrnrlon=-113,urcrnrlat=42.5,resolution='h')
        m.drawcoastlines()
        m.drawstates()
        m.drawcountries()
        # parallels = np.arange(30,45,5.) # make latitude lines 
        # meridians = np.arange(-125,-110,5.) # make longitude lines 
        # m.drawparallels(parallels,labels=[1,0,0,0],fontsize=9,alpha=0.4,color='gray')
        # m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=9,alpha=0.4,color='gray')
        m.drawlsmask(land_color='Linen',ocean_color='lightblue',lakes=True,resolution = 'h')
        m.drawrivers(linewidth=0.5, linestyle='solid', color='dodgerblue', zorder=1)
        # m.shadedrelief(scale=0.7,alpha=0.6) # add topography
        for i,key in enumerate(self.network_param.keys()):
            x,y=m(self.network_param[key]['lon'],self.network_param[key]['lat'])
            m.scatter(x,y,marker='o',alpha=0.7,s=self.network_param[key]['capacity_MW']/5,label=key+' ('+str(int(round(self.network_param[key]['capacity_MW'],0)))+' MW)')
        plt.legend(fontsize=9,loc=1)
        m.drawmapscale(-123,33,0,0,200,fontsize=8)
        plt.title('Modeled hydropower plants', loc='left', fontweight='bold')
        plt.tight_layout()
        plt.savefig('schematic/schematic.pdf',transparent=True)
        plt.close(fig)
        print('model schematics have been saved')

    def preprocess_LP(self,datadir='model/data_lp.csv',plot_benefit_curves=False):
        print('*******\ncreating LP network \n*******')

        from scipy import optimize # used to fit linear curve
   
        network_param = self.network_param
        price = self.price
        flow = self.flow
        network_conn = self.network_conn
        inf_bound = self.inf_bound
        spill_cost = self.spill_cost

        # create timestamps from indices
        index = list(flow.index)
        index.append('ENDING')

        # default parameter values
        def_price = 0
        def_a = 1
        def_lower_b = 0
        def_upper_b = inf_bound

        # an empty list to save network data        
        df_list = []
        # create links for intial storages
        df_list.append(['SUPERSOURCE','INITIAL',def_price,def_a,def_lower_b,def_upper_b])
        # create links for ending storages
        df_list.append(['ENDING','SUPERSINK',def_price,def_a,def_lower_b,def_upper_b])
        # write initial storage values
        for j in range(len(network_param.keys())):
            df_list.append(['INITIAL','stor_'+network_param.keys()[j]+'.'+str(index[0]),def_price,def_a,network_param[network_param.keys()[j]].loc['initial_storage (million m3)'],network_param[network_param.keys()[j]].loc['initial_storage (million m3)']])
        for i in range(len(index)-1):
            # create subsource links for each time-step from supersource
            df_list.append(['SUPERSOURCE','INFLOW.'+str(index[i]),def_price,def_a,def_lower_b,def_upper_b])
            # create subsink links for each time-step from supersink
            df_list.append(['DELTA.'+str(index[i]),'SUPERSINK',def_price,def_a,def_lower_b,def_upper_b])
            for j in range(len(network_param.keys())):
                # find best linear curve fit to nonlinear benefit curve
                fl = np.linspace(0,network_param[network_param.keys()[j]].loc['release_capacity (cms)'],15)
                st = np.linspace(network_param[network_param.keys()[j]].loc['deadpool (million m3)'],network_param[network_param.keys()[j]].loc['storage_capacity (million m3)'],15)
                stx, fly = np.meshgrid(st,fl)
                benefit_nonlinear = hydropower_rev(1,fly,stx,network_param[network_param.keys()[j]].loc['efficiency'],self.rho,self.g,
                                    network_param[network_param.keys()[j]].loc['a3'],
                                    network_param[network_param.keys()[j]].loc['a2'],
                                    network_param[network_param.keys()[j]].loc['a1'],
                                    network_param[network_param.keys()[j]].loc['c'],
                                    self.total_seconds[self.freq]/3600,
                                    price.iloc[i]
                                    )
                def fun(x): # maximize r2
                    benefit_linear = stx*x[0]+fly*x[1]
                    SSE = np.sum((benefit_nonlinear-benefit_linear)**2)
                    SST = np.sum((benefit_nonlinear-np.mean(benefit_nonlinear))**2)
                    r2 = 1-SSE/SST
                    return -r2
                x0 = np.array([1,1])
                res = optimize.fmin(fun,x0,disp=False)
                # print(-fun(res)) # r2 value
                unit_s = res[0] # slope for storage $/million m3
                unit_f = res[1] # slope for flow $/m3/s

                if plot_benefit_curves:
                    if i==0 and j==0:
                        from mpl_toolkits.mplot3d import Axes3D

                        benefit_linear = stx*unit_s+fly*unit_f

                        # linear model
                        fig = plt.figure()
                        ax = fig.gca(projection='3d')
                        # Plot the surface.
                        surf = ax.plot_surface(stx, fly, benefit_linear, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.99)
                        ax.set_xlabel('Storage million $m^3$')
                        ax.set_ylabel('Release $m^3/s$')
                        ax.set_zlabel('Revenue $')
                        plt.title('Linear model benefit curve',fontweight='bold')
                        plt.xlim([0,st.max()])
                        plt.ylim([0,fl.max()])
                        ax.set_zlim([0,max(benefit_linear.max(),benefit_nonlinear.max())])
                        plt.savefig('output_plots/lp.pdf',transparent=True)
                        plt.close(fig)

                        # nonlinear model
                        fig = plt.figure()
                        ax = fig.gca(projection='3d')
                        # Plot the surface.
                        surf = ax.plot_surface(stx, fly, benefit_nonlinear, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.99)
                        ax.set_xlabel('Storage million $m^3$')
                        ax.set_ylabel('Release $m^3/s$')
                        ax.set_zlabel('Revenue $')
                        plt.title('Nonlinear model benefit curve',fontweight='bold')
                        plt.xlim([0,st.max()])
                        plt.ylim([0,fl.max()])
                        ax.set_zlim([0,max(benefit_linear.max(),benefit_nonlinear.max())])
                        plt.savefig('output_plots/nlp.pdf',transparent=True)
                        plt.close(fig)

                        # residuals
                        fig = plt.figure()
                        ax = fig.gca(projection='3d')
                        # Plot the surface.
                        surf = ax.plot_surface(stx, fly, benefit_nonlinear-benefit_linear, cmap=cm.coolwarm,linewidth=0, antialiased=False)
                        ax.set_xlabel('Storage million $m^3$')
                        ax.set_ylabel('Release $m^3/s$')
                        ax.set_zlabel('Error $')
                        plt.title('Residuals curve (nonlinear - linear)',fontweight='bold')
                        plt.xlim([0,st.max()])
                        plt.ylim([0,fl.max()])
                        plt.savefig('output_plots/residuals.pdf',transparent=True)
                        plt.close(fig)
     
                # from supersource, create inflows to plants (constrained lb=ub)
                df_list.append(['INFLOW.'+str(index[i]),'stor_'+network_param.keys()[j]+'.'+str(index[i]),def_price,def_a,flow[self.CDEC_station_locator[network_param.keys()[j]]].iloc[i]*self.conv_fac,flow[self.CDEC_station_locator[network_param.keys()[j]]].iloc[i]*self.conv_fac])
                # write storage properties (ub=0 if no storage capacity)
                df_list.append(['stor_'+network_param.keys()[j]+'.'+str(index[i]),
                                str(index[i+1]) if str(index[i+1]) == 'ENDING' else 'stor_'+network_param.keys()[j]+'.'+str(index[i+1]),
                                unit_s,
                                def_a if self.freq == 'H' else network_param[network_param.keys()[j]].loc['evap_coeff'], # use default amplitude if hourly time-step
                                network_param[network_param.keys()[j]].loc['ending_storage (million m3)'] if str(index[i+1]) == 'ENDING' else network_param[network_param.keys()[j]].loc['deadpool (million m3)'],
                                network_param[network_param.keys()[j]].loc['ending_storage (million m3)'] if str(index[i+1]) == 'ENDING' else network_param[network_param.keys()[j]].loc['storage_capacity (million m3)']])
                # write turbine release links (energy prices are here)
                df_list.append(['stor_'+network_param.keys()[j]+'.'+str(index[i]),'flow_'+network_param.keys()[j]+'.'+str(index[i]),unit_f/self.conv_fac,def_a,def_lower_b,network_param[network_param.keys()[j]].loc['release_capacity (cms)']*self.conv_fac])
                # write spill links (penalties for spilling)
                df_list.append(['stor_'+network_param.keys()[j]+'.'+str(index[i]),'spill_'+network_param.keys()[j]+'.'+str(index[i]),spill_cost,def_a,def_lower_b,def_upper_b])
                # combine spills and turbine releases downstream
                df_list.append(['spill_'+network_param.keys()[j]+'.'+str(index[i]),'flow_'+network_param.keys()[j]+'.'+str(index[i]),def_price,def_a,def_lower_b,def_upper_b])
                # connect reservoir downstream flow nodes to sink or downstream reservoir
                connected = False
                i1 = network_param.keys()[j]
                for i2 in network_param.columns:
                    if [i1,i2] in network_conn:
                        connected = True
                        df_list.append(['flow_'+i1+'.'+str(index[i]),'stor_'+i2+'.'+str(index[i]),def_price,def_a,def_lower_b,def_upper_b])
                        break
                if not connected:
                    df_list.append(['flow_'+i1+'.'+str(index[i]),'DELTA.'+str(index[i]),def_price,def_a,def_lower_b,def_upper_b])
        datafile = pd.DataFrame(df_list,columns=['i','j','price','amplitude','lower_bound','upper_bound'])
        
        datafile.to_csv(datadir,index=False)
        print('linear model network data has been exported')

    def create_pyomo_NLP(self,datadir='model/data_nlp.csv',display_model=False):
        # data file containing parameters
        df = pd.read_csv(datadir)

        # links are from node_i to node_j
        df['link'] = df.i.map(str) + '_' + df.j.map(str)
        df.set_index('link', inplace=True)
        self.df = df

        # storage and release nodes
        self.nodes = pd.unique(df[['i','j']].values.ravel()).tolist()
        # storage and release links
        self.links = list(zip(df.i,df.j))

        print('creating NLP pyomo model')

        Model = ConcreteModel(name="Hydropower Model")

        # retrieve link parameters
        def init_params(p):
            return lambda Model,i,j: df.loc[str(i)+'_'+str(j)][p]

        # conversion factor for converting m3/s to m3
        convert = self.conv_fac

        # separate storage and flow links
        l_flow = []
        l_storage = []
        for link in self.links:
            if ('stor' in link[0] or 'INITIAL' in link[0]):
                if 'stor' in link[1] or 'ENDING' in link[1]:
                    l_storage.append(link)
                else:
                    l_flow.append(link)
            else:
                l_flow.append(link)

        self.l_flow = l_flow
        self.l_storage = l_storage

        Model.l_flow = Set(initialize=l_flow, doc='flow links')
        Model.l_stor = Set(initialize=l_storage, doc='storage links')

        Model.N = Set(initialize=self.nodes, doc='network nodes')
        Model.A = Set(within=Model.N*Model.N,initialize=self.links, ordered=True, doc='network links')

        Model.price = Param(Model.A, initialize=init_params('price'), doc='energy price ($/MWh)')
        Model.a3 = Param(Model.A, initialize=init_params('a3'), default=0, doc='polynomial parameter')
        Model.a2 = Param(Model.A, initialize=init_params('a2'), default=0, doc='polynomial parameter')
        Model.a1 = Param(Model.A, initialize=init_params('a1'), default=0, doc='polynomial parameter')
        Model.c = Param(Model.A, initialize=init_params('c'), default=0, doc='polynomial parameter')
        Model.efficiency = Param(Model.A, initialize=init_params('efficiency'), default=1, doc='efficiency')
        Model.amplitude = Param(Model.A, initialize=init_params('amplitude'), doc='reservoir evaporation coefficient')

        Model.l_f = Param(Model.l_flow, initialize=init_params('lower_bound'),mutable=True, doc='flow link lower bound')
        Model.u_f = Param(Model.l_flow, initialize=init_params('upper_bound'),mutable=True, doc='flow link upper bound')

        Model.l_s = Param(Model.l_stor, initialize=init_params('lower_bound'),mutable=True, doc='storage link lower bound')
        Model.u_s = Param(Model.l_stor, initialize=init_params('upper_bound'),mutable=True, doc='storage link upper bound')

        Model.source = Param(initialize='SUPERSOURCE', doc='super source node')
        Model.sink = Param(initialize='SUPERSINK', doc='super sink node')

        # find terminal (outgoing) nodes for flow links
        def NodesOut_init_flow(Model, node):
            retval = []
            for (i,j) in Model.l_flow:
                if i == node:
                    retval.append(j)
            return retval
        Model.NodesOut_flow = Set(Model.N, initialize=NodesOut_init_flow, doc='outgoing flow nodes')

        # find terminal (outgoing) nodes for storage links
        def NodesOut_init_stor(Model, node):
            retval = []
            for (i,j) in Model.l_stor:
                if i == node:
                    retval.append(j)
            return retval
        Model.NodesOut_stor = Set(Model.N, initialize=NodesOut_init_stor, doc='outgoing storage nodes')

        # find origin (incoming) nodes for flow links
        def NodesIn_init_flow(Model, node):
            retval = []
            for (i,j) in Model.l_flow:
                if j == node:
                    retval.append(i)
            return retval
        Model.NodesIn_flow = Set(Model.N, initialize=NodesIn_init_flow, doc='incoming flow nodes')

        # find origin (incoming) nodes for storage links
        def NodesIn_init_stor(Model, node):
            retval = []
            for (i,j) in Model.l_stor:
                if j == node:
                    retval.append(i)
            return retval
        Model.NodesIn_stor = Set(Model.N, initialize=NodesIn_init_stor, doc='incoming storage nodes')

        if self.warmstart:
            # flow decision variables
            Model.flow = Var(Model.l_flow,within=Reals, initialize=init_params('warmstart'), doc='flow decisions')
            # storage decision variables
            Model.storage = Var(Model.l_stor,within=Reals, initialize=init_params('warmstart'), doc='storage decisions')
            ### Declare all suffixes 
            # Ipopt bound multipliers (obtained from solution)
            Model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            Model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            # Ipopt bound multipliers (sent to solver)
            Model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            Model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
            # Obtain dual solutions from first solve and send to warm start
            Model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        else:
            # flow decision variables
            Model.flow = Var(Model.l_flow,within=Reals, doc='flow decisions')
            # storage decision variables
            Model.storage = Var(Model.l_stor,within=Reals, doc='storage decisions')
            # Create a 'dual' suffix component on the instance
            # so the solver plugin will know which suffixes to collect
            Model.dual = Suffix(direction=Suffix.IMPORT)

        # total number of hours in defined time-step
        num_hours = self.total_seconds[self.freq]/3600

        # Maximize total benefit
        def obj_fxn(Model):
            return sum(hydropower_rev(convert,Model.flow[i],Model.storage[j],Model.efficiency[j],self.rho,self.g,Model.a3[j],Model.a2[j],Model.a1[j],Model.c[j],num_hours,Model.price[i]) for i in Model.l_flow for j in Model.l_stor)+sum(Model.flow[i]*Model.price[i] for i in Model.l_flow)
        Model.obj = Objective(rule=obj_fxn, sense=maximize, doc='objective function')

        # Enforce an upper bound limit on the flow across each arc
        def limit_rule_upper_flow(Model, i,j):
            return Model.flow[i,j] <= Model.u_f[i,j]
        Model.limit_upper_flow = Constraint(Model.l_flow, rule=limit_rule_upper_flow)

        # Enforce an upper bound limit on the storage across each arc
        def limit_rule_upper_stor(Model, i,j):
            return Model.storage[i,j] <= Model.u_s[i,j]
        Model.limit_upper_stor = Constraint(Model.l_stor, rule=limit_rule_upper_stor)

        # Enforce a lower bound limit on the flow across each arc
        def limit_rule_lower_flow(Model, i,j):
            return Model.flow[i,j] >= Model.l_f[i,j]
        Model.limit_lower_flow = Constraint(Model.l_flow, rule=limit_rule_lower_flow)

        # Enforce a lower bound limit on the storage across each arc
        def limit_rule_lower_stor(Model, i,j):
            return Model.storage[i,j] >= Model.l_s[i,j]
        Model.limit_lower_stor = Constraint(Model.l_stor, rule=limit_rule_lower_stor)

        # enforce mass balance
        def MassBalance_rule(Model, node):
            if node in [value(Model.source), value(Model.sink)]:
                return Constraint.Skip
            inflow = sum(Model.flow[i,node] for i in Model.NodesIn_flow[node])+sum(Model.storage[i,node] for i in Model.NodesIn_stor[node])
            outflow = sum(Model.flow[node,j]/Model.amplitude[node,j] for j in Model.NodesOut_flow[node])+sum(Model.storage[node,j]/Model.amplitude[node,j] for j in Model.NodesOut_stor[node])
            return inflow == outflow
        Model.MassBalance = Constraint(Model.N, rule=MassBalance_rule)

        # print the model built before sending to solver
        if display_model:
            Model.pprint()

        self.Model = Model

    def solve_pyomo_NLP(self,solver='ipopt',stream_solver=False,display_model_out=False,display_raw_results=False,max_iter=3000,tol=1e-08,mu_init=1e-1,max_cpu_time=1e+06,constr_viol_tol=0.0001,acceptable_constr_viol_tol=1e-6):
        print('solving NLP problem')

        Model = self.Model

        # specify solver
        opt = SolverFactory(solver,solver_io ='nl')
        if solver == 'ipopt':
            # some solver specific options. for more options type in command line 'ipopt --print-options'
            opt.options['max_iter'] = max_iter # maximum number of iterations
            opt.options['tol'] = tol # Desired convergence tolerance
            opt.options['constr_viol_tol'] = constr_viol_tol # Desired threshold for the constraint violation
            opt.options['max_cpu_time'] = max_cpu_time # Maximum number of CPU seconds
            opt.options['acceptable_constr_viol_tol'] = acceptable_constr_viol_tol # "Acceptance" threshold for the constraint violation
            opt.options['mu_init'] = mu_init # Initial value for the barrier parameter         

        # solve the model
        self.results = opt.solve(Model, tee=stream_solver)

        # save stats
        with open('summary_nlp_model.txt', 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerow([self.results])

        if self.warmstart and solver == 'ipopt':
            ### Set Ipopt options for warm-start
            # The current values on the ipopt_zU_out and
            # ipopt_zL_out suffixes will be used as initial 
            # conditions for the bound multipliers to solve the new problem
            Model.ipopt_zL_in.update(Model.ipopt_zL_out)
            Model.ipopt_zU_in.update(Model.ipopt_zU_out)
            # parameter source: https://www.gams.com/latest/docs/S_IPOPT.html
            opt.options['warm_start_init_point'] = 'yes'
            opt.options['warm_start_bound_push'] = 1e-9
            opt.options['warm_start_mult_bound_push'] = 1e-9
            print("WARM-STARTED SOLVE")
            # The solver plugin will scan the model for all active suffixes
            # valid for importing, which it will store into the results object
            self.results = opt.solve(Model, tee=stream_solver)

            # save stats
            with open('summary_nlp_model_ws.txt', 'w') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerow([self.results])

        # solver status
        print ('The solver returned a status of: '+str(self.results.solver.termination_condition))

        # print the model after solver solves
        if display_model_out:
            Model.pprint()

        # # send results to stdout
        # Model.solutions.store_to(self.results) # summary and variables
        # self.results.write() # display results

        # # display output
        if display_raw_results:
            Model.display()

        self.Model_type = 'nonlinear'

    def create_pyomo_LP(self,datadir='model/data_lp.csv',display_model=False):
        # data file containing parameters
        df = pd.read_csv(datadir)

        # links are from node_i to node_j
        df['link'] = df.i.map(str) + '_' + df.j.map(str)
        df.set_index('link', inplace=True)
        self.df = df

        # storage and release nodes
        self.nodes = pd.unique(df[['i','j']].values.ravel()).tolist()
        # storage and release links
        self.links = list(zip(df.i,df.j))

        print('creating LP pyomo model')

        Model = ConcreteModel(name="Hydropower LP Model")

        # retrieve link parameters
        def init_params(p):
            return lambda Model,i,j: df.loc[str(i)+'_'+str(j)][p]

        # separate storage and flow links
        l_flow = []
        l_storage = []
        for link in self.links:
            if ('stor' in link[0] or 'INITIAL' in link[0]):
                if 'stor' in link[1] or 'ENDING' in link[1]:
                    l_storage.append(link)
                else:
                    l_flow.append(link)
            else:
                l_flow.append(link)

        self.l_flow = l_flow
        self.l_storage = l_storage

        Model.l_flow = Set(initialize=l_flow, doc='flow links')
        Model.l_stor = Set(initialize=l_storage, doc='storage links')

        Model.N = Set(initialize=self.nodes, doc='network nodes')
        Model.A = Set(within=Model.N*Model.N,initialize=self.links, ordered=True, doc='network links')
        
        Model.price = Param(Model.A, initialize=init_params('price'), doc='energy price ($/MWh)')
        Model.amplitude = Param(Model.A, initialize=init_params('amplitude'), doc='reservoir evaporation coefficient')
        Model.l = Param(Model.A, initialize=init_params('lower_bound'),mutable=True, doc='flow link lower bound')
        Model.u = Param(Model.A, initialize=init_params('upper_bound'),mutable=True, doc='flow link upper bound')

        Model.source = Param(initialize='SUPERSOURCE', doc='super source node')
        Model.sink = Param(initialize='SUPERSINK', doc='super sink node')

        # find terminal (outgoing) nodes for flow links
        def NodesOut_init(Model, node):
            retval = []
            for (i,j) in Model.A:
                if i == node:
                    retval.append(j)
            return retval
        Model.NodesOut = Set(Model.N, initialize=NodesOut_init, doc='outgoing nodes')

        # find origin (incoming) nodes for flow links
        def NodesIn_init(Model, node):
            retval = []
            for (i,j) in Model.A:
                if j == node:
                    retval.append(i)
            return retval
        Model.NodesIn = Set(Model.N, initialize=NodesIn_init, doc='incoming nodes')

        # decision variables
        Model.X = Var(Model.A, within=Reals)
        # Create a 'dual' suffix component on the instance
        # so the solver plugin will know which suffixes to collect
        Model.dual = Suffix(direction=Suffix.IMPORT)

        # Maximize total benefit
        def obj_fxn(Model):
            return sum(Model.X[i,j]*Model.price[i,j] for (i,j) in Model.A)
        Model.obj = Objective(rule=obj_fxn, sense=maximize, doc='objective function')

        # Enforce an upper bound limit
        def limit_rule_upper(Model, i,j):
            return Model.X[i,j] <= Model.u[i,j]
        Model.limit_upper = Constraint(Model.A, rule=limit_rule_upper)

        # Enforce a lower bound limit on the flow across each arc
        def limit_rule_lower(Model, i,j):
            return Model.X[i,j] >= Model.l[i,j]
        Model.limit_lower = Constraint(Model.A, rule=limit_rule_lower)

        # enforce mass balance
        def MassBalance_rule(Model, node):
            if node in [value(Model.source), value(Model.sink)]:
                return Constraint.Skip
            inflow = sum(Model.X[i,node] for i in Model.NodesIn[node])
            outflow = sum(Model.X[node,j]/Model.amplitude[node,j] for j in Model.NodesOut[node])
            return inflow == outflow
        Model.MassBalance = Constraint(Model.N, rule=MassBalance_rule)

        # print the model built before sending to solver
        if display_model==True:
            Model.pprint()

        self.Model = Model

    def solve_pyomo_LP(self,solver='glpk',stream_solver=False,display_model_out=False,display_raw_results=False):
        print('solving LP problem')

        Model = self.Model

        # specify solver
        opt = SolverFactory(solver)

        # solve the model
        self.results = opt.solve(Model, tee=stream_solver)

        # solver status
        print ('The solver returned a status of: '+str(self.results.solver.termination_condition))

        # print the model after solver solves
        if display_model_out:
            Model.pprint()

        # # send results to stdout
        # Model.solutions.store_to(self.results) # summary and variables
        # self.results.write() # display results

        # save stats
        with open('summary_lp_model.txt', 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerow([self.results])

        # # display output
        if display_raw_results:
            Model.display()

        self.Model_type = 'linear'

    def postprocess(self,save_path=None):

        Model = self.Model

        # if directory to save model outputs do not exist, create one
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        # sort combined string and alphanumeric time keys nicely
        def sorted_nicely(l):
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(l, key = alphanum_key)

        # postprocess results and save
        def save_dict_as_csv(data, filename):
            node_keys = sorted_nicely(data.keys())
            time_keys = sorted_nicely(data[node_keys[0]].keys())

            dff = []
            header = ['date'] + node_keys

            for t in time_keys:
                row = [t]
                for k in node_keys:
                    if t in data[k] and data[k][t] is not None:
                        row.append(data[k][t])
                    else:
                        row.append(0.0)
                dff.append(row)
            pd.DataFrame(dff,columns=header).to_csv(filename,index=False)

        def dict_get(D, k1, k2, default = 0.0):
            if k1 in D and k2 in D[k1]:
                return D[k1][k2]
            else:
                return default

        def dict_insert(D, k1, k2, v):
            if k1 not in D:
                D[k1] = {k2: v}
            elif k2 not in D[k1]:
                D[k1][k2] = v
            else:
                raise ValueError('Keys [%s][%s] already exist in dictionary' % (k1,k2))

        # flow (F), storage (S), dual (D)
        F,S = {}, {}
        D_up_f,D_lo_f,D_up_s,D_lo_s,D_node = {}, {}, {}, {}, {}

        unsorted_f = []
        # get flow values
        for link in self.l_flow:
            if self.Model_type == 'nonlinear':
                f = Model.flow[link].value/self.conv_fac if link in Model.flow else 0.0
                d_lower_f = Model.dual[Model.limit_lower_flow[link]] if link in Model.limit_lower_flow else 0.0
                d_upper_f = Model.dual[Model.limit_upper_flow[link]] if link in Model.limit_upper_flow else 0.0
            else: # linear model
                f = Model.X[link].value/self.conv_fac if link in Model.X else 0.0
                d_lower_f = Model.dual[Model.limit_lower[link]] if link in Model.limit_lower else 0.0
                d_upper_f = Model.dual[Model.limit_upper[link]] if link in Model.limit_upper else 0.0
            unsorted_f.append([link,f*self.conv_fac])
            
            if link[0].split('.')[0]=='SOURCE' or link[1].split('.')[0] =='SINK':
                continue
            if '.' in link[0] and '.' in link[1]:
                n1,t1 = link[0].split('.')
                n2,t2 = link[1].split('.')
                key = n1 + '-' + n2
                dict_insert(F, key, t1, f)
                dict_insert(D_up_f, key, t1, d_lower_f)
                dict_insert(D_lo_f, key, t1, d_upper_f)
        unsorted_f=pd.DataFrame(unsorted_f,columns=['link','value']).to_csv(save_path+'/unsorted_f.csv',index=False)
        
        unsorted_s = []
        # get storage values
        for link in self.l_storage:
            if self.Model_type == 'nonlinear':
                s = Model.storage[link].value if link in Model.storage else 0.0
                d_lower_s = Model.dual[Model.limit_lower_stor[link]] if link in Model.limit_lower_stor else 0.0
                d_upper_s = Model.dual[Model.limit_upper_stor[link]] if link in Model.limit_upper_stor else 0.0
            else: # linear model
                s = Model.X[link].value if link in Model.X else 0.0
                d_lower_s = Model.dual[Model.limit_lower[link]] if link in Model.limit_lower else 0.0
                d_upper_s = Model.dual[Model.limit_upper[link]] if link in Model.limit_upper else 0.0
            unsorted_s.append([link,s])
            
            if link[0].split('.')[0]=='INITIAL':
                continue
            if ('.' in link[0] and '.' in link[1]) or ('.' in link[0] and 'ENDING' in link[1]):
                n1,t1 = link[0].split('.')
                key = n1
                dict_insert(S, key, t1, s)
                dict_insert(D_up_s, key, t1, d_lower_s)
                dict_insert(D_lo_s, key, t1, d_upper_s)
        unsorted_s=pd.DataFrame(unsorted_s,columns=['link','value']).to_csv(save_path+'/unsorted_s.csv',index=False)
        
        # get dual values for nodes (mass balance)
        for node in self.nodes:
            if '.' in node:
                n3,t3 = node.split('.')
                key = n3
                d3 = Model.dual[Model.MassBalance[node]] if node in Model.MassBalance else 0.0
                dict_insert(D_node, key, t3, d3)

        things_to_save = [
                            (F, 'flow_cms'), 
                            (S, 'storage_million_m3'), 
                            (D_up_f, 'dual_upper_flow'), 
                            (D_lo_f, 'dual_lower_flow'), 
                            (D_up_s, 'dual_upper_storage'), 
                            (D_lo_s, 'dual_lower_storage'),
                            (D_node, 'dual_node'),
                            ]

        # save results
        for data,name in things_to_save:
            save_dict_as_csv(data, save_path+'/'+name+'.csv')

        # calculate power and generation
        flow = pd.read_csv(save_path+'/flow_cms.csv',index_col=0,header=0)
        storage = pd.read_csv(save_path+'/storage_million_m3.csv',index_col=0,header=0)

        network_param = self.network_param
        power = pd.DataFrame(index=storage.index)
        generation = pd.DataFrame(index=storage.index)
        revenue = pd.DataFrame(index=storage.index)
        for plant in self.plants:
            pw = np.array(hydropower_rev(1,flow['stor_'+plant+'-'+'flow_'+plant].values,storage['stor_'+plant].values,network_param[plant]['efficiency'],self.rho,self.g,network_param[plant]['a3'],network_param[plant]['a2'],network_param[plant]['a1'],network_param[plant]['c'],1,1))
            # power cannot exceed power capacity
            pw[pw > self.network_param[plant].loc['capacity_MW']] = self.network_param[plant].loc['capacity_MW']
            power[plant]=pw
            generation[plant]=power[plant]*self.total_seconds[self.freq]/3600
            revenue[plant] = generation[plant]*self.price.values
        power.to_csv(save_path+'/power_MW.csv')
        generation.to_csv(save_path+'/generation_MWh.csv')
        revenue.to_csv(save_path+'/revenue_$.csv')
        print('results are saved to output folder \n*******')

