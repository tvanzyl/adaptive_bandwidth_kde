"""
A different smoothing parameter for each variable generally gives sufficient control.
A full bivariate normal kernel may  be used  in  special circumstances, effectively
adding one additional smoothing parameter in the form of the correlation coefficient.
However, an equivalent estimate may be obtained by rotating the data so that the
correlation in the kernel vanishes, so that the product kernel may be used on the
transformed data.
http://download.springer.com/static/pdf/998/chp%253A10.1007%252F978-3-642-21551-3_19.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Fchapter%2F10.1007%2F978-3-642-21551-3_19&token2=exp=1459348350~acl=%2Fstatic%2Fpdf%2F998%2Fchp%25253A10.1007%25252F978-3-642-21551-3_19.pdf%3ForiginUrl%3Dhttp%253A%252F%252Flink.springer.com%252Fchapter%252F10.1007%252F978-3-642-21551-3_19*~hmac=6bb24bd85f05d3889caf24b7b1257f9cf4499f963e7a0adb64abf8e1f164bad2
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.3682&rep=rep1&type=pdf (pg 4.)

However we need local rotations, for adaptive kernels

However, Wand and Jones (1993) urge
caution when using these approaches as they are not guaranteed to give the correct
transformation or rotation to achieve the gains possible by using the full smoothing
matrix. Many authors have noted that for most densities, in particular unimodal ones,
allowing different amounts of smoothing for each dimension (the product kernel
estimator,H2) is adequate. With more complex densities, especially multimodal
ones, the situation is less clear, although rotations can help if the structure of the
distribution can be aligned with the coordinate axis (Wand and Jones, 1993).
http://private.igf.edu.pl/~jnn/Literatura_tematu/Sain_2002.pdf
"""

"""
Projection Persuit
GMM (EM and CV)
Bayesian Networks 


Ref Style Harvard

"""

import samplepoint
import mvn
import balloon
import mlloo
import visualise
import data
import partition
import cluster
import bayesian
import pandas as pd

from pandas import DataFrame
from tabulate import tabulate
from numpy import array, random
from design import *

import time

random.seed(11235)
#random.seed(1)

estimators = {
##SAMPLEPOINT EXPERIMENTS
#"cv_ls H2": mvn.GlobalKDE('cv_ls', covariance='H2'),
#"cv_ls H3": mvn.GlobalKDE('cv_ls', covariance='H3'),
#"rule-of-thumb terrel H2": mvn.GlobalKDE('rule-of-thumb', covariance='H2'), #H3 best
#"rule-of-thumb terrel H3": mvn.GlobalKDE('rule-of-thumb', covariance='H3'), #H3 best
#"samplepoint silverman H2 backsub": samplepoint.SamplePointKDE(covariance='H2'), #A classic
#"samplepoint silverman H3 backsub": samplepoint.SamplePointKDE(covariance='H3'), #A classic
#"samplepoint silverman H2 cv_ls": samplepoint.SamplePointKDE(covariance='H2', optimiser='cv_ls'), #H3 and cv_ls best
"samplepoint silverman H3 cv_ls": samplepoint.SamplePointKDE(covariance='H3', optimiser='cv_ls'), #H3 and cv_ls best
#
####Breiman will always be in H1 since its a knn technique
"samplepoint breiman H1 cv_ls kung": samplepoint.SamplePointKDE(alpha='1/d', pilotfactor='breiman', k='kung', covariance='H2', optimiser='cv_ls'),
#"samplepoint breiman H1 cv_ls hansen": samplepoint.SamplePointKDE(alpha='1/d', pilotfactor='breiman', k='hansen', covariance='H2', optimiser='cv_ls'),
#"samplepoint breiman H1 cv_ls sqrt": samplepoint.SamplePointKDE(alpha='1/d', pilotfactor='breiman', k='sqrt', covariance='H2', optimiser='cv_ls'),
#
####Wu CLUSTERING will always be in H2
"samplepoint wu H2 cv_ls": samplepoint.SamplePointKDE(pilotfactor='wu', covariance='H2', optimiser='cv_ls'), #No H3
#
####BALLOON EXPERIMENTS
#"balloon kung-knn H1 %s"%'loo_cv': balloon.BalloonKDE(k='loo_cv', balloon='kung-knn'),
#"balloon kung-knn H1 kung": balloon.BalloonKDE(k='kung', balloon='kung-knn'),
#"balloon kung-knn H1 hansen": balloon.BalloonKDE(k='hansen', balloon='kung-knn'),
#"balloon kung-knn H1 sqrt": balloon.BalloonKDE(k='sqrt', balloon='kung-knn'),
#
#"balloon biau-knn H1 %s"%'loo_cv': balloon.BalloonKDE(k='loo_cv', balloon='biau-knn'),
#"balloon biau-knn H3 kung": balloon.BalloonKDE(k='kung', balloon='biau-knn'),
#"balloon biau-knn H3 hansen": balloon.BalloonKDE(k='hansen', balloon='biau-knn'),
#"balloon biau-knn H3 sqrt": balloon.BalloonKDE(k='sqrt', balloon='biau-knn'),
#
#"balloon terrel-knn H3 %s"%'loo_cv': balloon.BalloonKDE(k='loo_cv', balloon='terrel-knn'),
#"balloon terrel-knn H3 kung": balloon.BalloonKDE(k='kung', balloon='terrel-knn'),
#"balloon terrel-knn H3 hansen": balloon.BalloonKDE(k='hansen', balloon='terrel-knn'),
"balloon terrel-knn H3 sqrt": balloon.BalloonKDE(k='sqrt', balloon='terrel-knn'),
#
#"balloon loftsgaarden-knn H1 %s"%'loo_cv': balloon.BalloonKDE(k='loo_cv', balloon='loftsgaarden-knn'),
#"balloon loftsgaarden-knn H1 kung": balloon.BalloonKDE(k='kung', balloon='loftsgaarden-knn'),
"balloon loftsgaarden-knn H1 hansen": balloon.BalloonKDE(k='hansen', balloon='loftsgaarden-knn'),
#"balloon loftsgaarden-knn H1 sqrt": balloon.BalloonKDE(k='sqrt', balloon='loftsgaarden-knn'),
#
#"balloon loftsgaarden-kernel-knn H1 %s"%'loo_cv': balloon.BalloonKDE(k='loo_cv', balloon='loftsgaarden-kernel-knn'),
#"balloon loftsgaarden-kernel-knn H1 kung": balloon.BalloonKDE(k='kung', balloon='loftsgaarden-kernel-knn',),
#"balloon loftsgaarden-kernel-knn H1 hansen": balloon.BalloonKDE(k='hansen', balloon='loftsgaarden-kernel-knn'),
"balloon loftsgaarden-kernel-knn H1 sqrt": balloon.BalloonKDE(k='sqrt', balloon='loftsgaarden-kernel-knn'),
#
####BAYESIAN
"balloon lima-quadratic H3": bayesian.BayesianKDE(bayes='lima_quadratic'),
#"balloon lima-entropy H3": bayesian.BayesianKDE(bayes='lima_entropy'),
#
####MAXIMUM LIKELIHOOD
"mlloo vdwalt H2":mlloo.LikelihoodKDE(regularizer='vdwalt', maxit=1),
#"mlloo barnard H2":mlloo.LikelihoodKDE(regularizer='barnard', maxit=1),

###LOO ML K
#"balloon kung-knn H1 %s"%'loo_ml': balloon.BalloonKDE(k='loo_ml', balloon='kung-knn'),
#"balloon kung-knn H1 %s"%'loo_cv': balloon.BalloonKDE(k='loo_cv', balloon='kung-knn'),
#"balloon kung-knn H1 kung": balloon.BalloonKDE(k='kung', balloon='kung-knn'),
#"balloon kung-knn H1 hansen": balloon.BalloonKDE(k='hansen', balloon='kung-knn'),
#"balloon kung-knn H1 sqrt": balloon.BalloonKDE(k='sqrt', balloon='kung-knn'),

#"balloon biau-knn H1 %s"%'loo_ml': balloon.BalloonKDE(k='loo_ml', balloon='biau-knn'),
#"balloon biau-knn H1 %s"%'loo_cv': balloon.BalloonKDE(k='loo_cv', balloon='biau-knn'),
#"balloon biau-knn H3 kung": balloon.BalloonKDE(k='kung', balloon='biau-knn'),
#"balloon biau-knn H3 hansen": balloon.BalloonKDE(k='hansen', balloon='biau-knn'),
#"balloon biau-knn H3 sqrt": balloon.BalloonKDE(k='sqrt', balloon='biau-knn'),
#
#"balloon terrel-knn H3 %s"%'loo_ml': balloon.BalloonKDE(k='loo_ml', balloon='terrel-knn'),
#"balloon terrel-knn H3 %s"%'loo_cv': balloon.BalloonKDE(k='loo_cv', balloon='terrel-knn'),
#"balloon terrel-knn H3 %s"%'loo_cv': balloon.BalloonKDE(k=40, balloon='terrel-knn'),
#"balloon terrel-knn H3 kung": balloon.BalloonKDE(k='kung', balloon='terrel-knn'),
#"balloon terrel-knn H3 hansen": balloon.BalloonKDE(k='hansen', balloon='terrel-knn'),
#"balloon terrel-knn H3 sqrt": balloon.BalloonKDE(k='sqrt', balloon='terrel-knn'),
#
#"balloon loftsgaarden-knn H1 %s"%'loo_ml': balloon.BalloonKDE(k='loo_ml', balloon='loftsgaarden-knn'),
#"balloon loftsgaarden-knn H1 %s"%'loo_cv': balloon.BalloonKDE(k='loo_cv', balloon='loftsgaarden-knn'),
#"balloon loftsgaarden-knn H1 kung": balloon.BalloonKDE(k='kung', balloon='loftsgaarden-knn'),
#"balloon loftsgaarden-knn H1 hansen": balloon.BalloonKDE(k='hansen', balloon='loftsgaarden-knn'),
#"balloon loftsgaarden-knn H1 sqrt": balloon.BalloonKDE(k='sqrt', balloon='loftsgaarden-knn'),
#
#"balloon loftsgaarden-kernel-knn H1 %s"%'loo_ml': balloon.BalloonKDE(k='loo_ml', balloon='loftsgaarden-kernel-knn'),
#"balloon loftsgaarden-kernel-knn H1 %s"%'loo_cv': balloon.BalloonKDE(k='loo_cv', balloon='loftsgaarden-kernel-knn'),
#"balloon loftsgaarden-kernel-knn H1 kung": balloon.BalloonKDE(k='kung', balloon='loftsgaarden-kernel-knn',),
#"balloon loftsgaarden-kernel-knn H1 hansen": balloon.BalloonKDE(k='hansen', balloon='loftsgaarden-kernel-knn'),
#"balloon loftsgaarden-kernel-knn H1 sqrt": balloon.BalloonKDE(k='sqrt', balloon='loftsgaarden-kernel-knn'),

}

####Excluded Experiments
##"samplepoint silverman H2 cv_ls_ndim": samplepoint.SamplePointKDE(covariance='H2', optimiser='cv_ls_ndim'), #H3 and cv_ls best
##"samplepoint silverman H3 cv_ls_ndim": samplepoint.SamplePointKDE(covariance='H3', optimiser='cv_ls_ndim'), #H3 and cv_ls best
##"samplepoint wu H2 cv_ls_ndim": samplepoint.SamplePointKDE(pilotfactor='wu', covariance='H2', optimiser='cv_ls_ndim'),
##"samplepoint wu H3 cv_ls_ndim": samplepoint.SamplePointKDE(pilotfactor='wu', covariance='H3', optimiser='cv_ls_ndim'),
##"partition sain cv_ls":partition.PartitionKDE(covariance='H2', partitions=20), #state of the art
##"partition sain rule-of-thumb":partition.PartitionKDE(pilot='rule-of-thumb', covariance='H3'),
##"balloon hall-pilot-knn H3 kung": balloon.BalloonKDE(k='kung', balloon='hall-pilot-knn'),
##"balloon hall-pilot-knn H3 hansen": balloon.BalloonKDE(k='hansen', balloon='hall-pilot-knn'),
##"balloon hall-pilot-knn H3 sqrt": balloon.BalloonKDE(k='sqrt', balloon='hall-pilot-knn'),
##"balloon loftsgaarden-kernel-knn H1 kung 2 sigma": balloon.BalloonKDE(k='kung', balloon='loftsgaarden-kernel-knn', percentile=0.9544),
##"balloon loftsgaarden-kernel-knn H1 hansen 2 sigma": balloon.BalloonKDE(k='hansen', balloon='loftsgaarden-kernel-knn', percentile=0.9544),
##"balloon loftsgaarden-kernel-knn H1 sqrt 2 sigma": balloon.BalloonKDE(k='sqrt', balloon='loftsgaarden-kernel-knn', percentile=0.9544),
##"balloon loftsgaarden-kernel-knn H1 kung 3 sigma": balloon.BalloonKDE(k='kung', balloon='loftsgaarden-kernel-knn', percentile=0.9973),
##"balloon loftsgaarden-kernel-knn H1 hansen 3 sigma": balloon.BalloonKDE(k='hansen', balloon='loftsgaarden-kernel-knn', percentile=0.9973),
##"balloon loftsgaarden-kernel-knn H1 sqrt 3 sigma": balloon.BalloonKDE(k='sqrt', balloon='loftsgaarden-kernel-knn', percentile=0.9973),
##"hybrid mittal H2 kung": balloon.BalloonKDE(k='kung', balloon='mittal'),
##"hybrid mittal H2 hansen": balloon.BalloonKDE(k='hansen', balloon='mittal'),
##"hybrid mittal H2 sqrt": balloon.BalloonKDE(k='sqrt', balloon='mittal'),


##FIND BEST K
#for k in range(2,20,1):
#    estimators["balloon terrel-knn H3 %s"%k] = balloon.BalloonKDE(k=k, balloon='terrel-knn')
##    estimators["balloon biau-ellipse-knn H3 %s"%k] = balloon.BalloonKDE(k=k, balloon='biau-ellipse-knn')
#    estimators["balloon loftsgaarden-knn H1 %s"%k] = balloon.BalloonKDE(k=k, balloon='loftsgaarden-knn')
#    estimators["balloon kung-knn H1 %s"%k] = balloon.BalloonKDE(k=k, balloon='kung-knn')
#    estimators["balloon biau-knn H1 %s"%k] = balloon.BalloonKDE(k=k, balloon='biau-knn')
#    estimators["balloon loftsgaarden-kernel-knn H1 %s"%k] = balloon.BalloonKDE(k=k, balloon='loftsgaarden-kernel-knn')

#NEW BEST EVER
#for k in range(2,10,1):
#    estimators["balloon biau-knn H1 %s"%k] = balloon.BalloonKDE(k=k, balloon='biau-knn')
#estimators["balloon biau-ellipse-knn H3 %s"%'loo_cv'] = balloon.BalloonKDE(k='loo_cv', balloon='biau-ellipse-knn')

results = {}
models = {}
#repeats = 30
repeats = 1

#Sample Sizes
#https://stats.stackexchange.com/questions/76948/what-is-the-minimum-number-of-data-points-required-for-kernel-density-estimation

experiments = [
#(Experiment1, {'n_training':110, 'n_test':2000}),
#(Experiment2, {'n_training':110, 'n_test':2000}),
#(Experiment3, {'n_training':110, 'n_test':2000}),
#(Experiment4, {'n_training':110, 'n_test':2000}),
#(Experiment5, {'n_training':110, 'n_test':2000}),

###Experiment 6 for dimensionality
#(Experiment6, {'dimensions':4,  'n_training':2000, 'n_test':2000}),
#(Experiment6, {'dimensions':6,  'n_training':3000, 'n_test':2000}),
#(Experiment6, {'dimensions':8,  'n_training':4000, 'n_test':2000}),
#(Experiment6, {'dimensions':10, 'n_training':5000, 'n_test':2000}),
#(Experiment6, {'dimensions':12, 'n_training':6000, 'n_test':2000}),
#(Experiment6, {'dimensions':14, 'n_training':7000, 'n_test':2000}),

##Experiment 7 for dimensionality
#(Experiment7, {'dimensions':4,  'n_training':2000, 'n_test':2000}),
#(Experiment7, {'dimensions':6,  'n_training':3000, 'n_test':2000}),
#(Experiment7, {'dimensions':8,  'n_training':4000, 'n_test':2000}),
#(Experiment7, {'dimensions':10, 'n_training':5000, 'n_test':2000}),
#(Experiment7, {'dimensions':12, 'n_training':6000, 'n_test':2000}),
#(Experiment7, {'dimensions':14, 'n_training':7000, 'n_test':2000}),

###Experiment 7 for sample size
#(Experiment7, {'dimensions':5,  'n_training':int(800*0.4), 'n_test':2000}),
#(Experiment7, {'dimensions':5,  'n_training':int(800*0.6), 'n_test':2000}),
#(Experiment7, {'dimensions':5,  'n_training':int(800*0.8), 'n_test':2000}),
#(Experiment7, {'dimensions':5,  'n_training':int(800*1.0), 'n_test':2000}),
#(Experiment7, {'dimensions':5,  'n_training':int(800*1.2), 'n_test':2000}),
#(Experiment7, {'dimensions':5,  'n_training':int(800*1.4), 'n_test':2000}),
#(Experiment7, {'dimensions':5,  'n_training':int(800*1.6), 'n_test':2000}),

###Experiment 7 for sample size
#(Experiment7, {'dimensions':6,  'n_training':int(2100*0.4), 'n_test':2000}),
#(Experiment7, {'dimensions':6,  'n_training':int(2100*0.6), 'n_test':2000}),
#(Experiment7, {'dimensions':6,  'n_training':int(2100*0.8), 'n_test':2000}),
#(Experiment7, {'dimensions':6,  'n_training':int(2100*1.0), 'n_test':2000}),
#(Experiment7, {'dimensions':6,  'n_training':int(2100*1.2), 'n_test':2000}),
#(Experiment7, {'dimensions':6,  'n_training':int(2100*1.4), 'n_test':2000}),
#(Experiment7, {'dimensions':6,  'n_training':int(2100*1.6), 'n_test':2000}),

###Experiment 8 for sample size
#(Experiment8, {'n_training':int(350*0.4), 'n_test':2000}),
#(Experiment8, {'n_training':int(350*0.6), 'n_test':2000}),
#(Experiment8, {'n_training':int(350*0.8), 'n_test':2000}),
#(Experiment8, {'n_training':int(350*1.0), 'n_test':2000}),
#(Experiment8, {'n_training':int(350*1.2), 'n_test':2000}),
#(Experiment8, {'n_training':int(350*1.4), 'n_test':2000}),
#(Experiment8, {'n_training':int(350*1.6), 'n_test':2000}),

###Experiment D for complexity
#(ExperimentD, {'modes':2,'dimensions':4, 'n_training':350, 'n_test':2000}),
#(ExperimentD, {'modes':3,'dimensions':4, 'n_training':350, 'n_test':2000}),
#(ExperimentD, {'modes':4,'dimensions':4, 'n_training':350, 'n_test':2000}),
#(ExperimentD, {'modes':5,'dimensions':4, 'n_training':350, 'n_test':2000}),
#(ExperimentD, {'modes':6,'dimensions':4, 'n_training':350, 'n_test':2000}),

#(ExperimentD, {'modes':2,'dimensions':5, 'n_training':800, 'n_test':2000}),
#(ExperimentD, {'modes':3,'dimensions':5, 'n_training':800, 'n_test':2000}),
#(ExperimentD, {'modes':4,'dimensions':5, 'n_training':800, 'n_test':2000}),
#(ExperimentD, {'modes':5,'dimensions':5, 'n_training':800, 'n_test':2000}),
#(ExperimentD, {'modes':6,'dimensions':5, 'n_training':800, 'n_test':2000}),

#(ExperimentD, {'modes':2,'dimensions':6, 'n_training':2100, 'n_test':2000}),
#(ExperimentD, {'modes':3,'dimensions':6, 'n_training':2100, 'n_test':2000}),
#(ExperimentD, {'modes':4,'dimensions':6, 'n_training':2100, 'n_test':2000}),
#(ExperimentD, {'modes':5,'dimensions':6, 'n_training':2100, 'n_test':2000}),
#(ExperimentD, {'modes':6,'dimensions':6, 'n_training':2100, 'n_test':2000}),

###Experiment E for complexity
(ExperimentE, {'dimensions':2, 'n_training':110,  'n_test':2000}),
#(ExperimentE, {'dimensions':3, 'n_training':175,  'n_test':2000}),
#(ExperimentE, {'dimensions':4, 'n_training':350,  'n_test':2000}),
#(ExperimentE, {'dimensions':5, 'n_training':800,  'n_test':2000}),
#(ExperimentE, {'dimensions':6, 'n_training':2100, 'n_test':2000}),
#(ExperimentE, {'dimensions':7, 'n_training':5700, 'n_test':2000}),

###Experiment E for complexity dimensionality
#(ExperimentE, {'dimensions':4,  'n_training':2000, 'n_test':2000}),
#(ExperimentE, {'dimensions':6,  'n_training':3000, 'n_test':2000}),
#(ExperimentE, {'dimensions':8,  'n_training':4000, 'n_test':2000}),
#(ExperimentE, {'dimensions':10, 'n_training':5000, 'n_test':2000}),
#(ExperimentE, {'dimensions':12, 'n_training':6000, 'n_test':2000}),
#(ExperimentE, {'dimensions':14, 'n_training':7000, 'n_test':2000}),

#8   19500
#9   74000
#10 299149 

#MNIST 60000
##Experiment E2 for complexity
#(ExperimentE, {'dimensions':2, 'n_training':110*2,  'n_test':2000}),
#(ExperimentE, {'dimensions':3, 'n_training':175*2,  'n_test':2000}),
#(ExperimentE, {'dimensions':4, 'n_training':350*2,  'n_test':2000}),
#(ExperimentE, {'dimensions':5, 'n_training':800*2,  'n_test':2000}),
#(ExperimentE, {'dimensions':6, 'n_training':2100*2, 'n_test':2000}),
#(ExperimentE, {'dimensions':7, 'n_training':5700*2, 'n_test':2000}),
]

update_results = False

for experiment_class, kwargs in experiments:
    dfResults = DataFrame([], index=range(repeats), 
    columns=['Experiment', 'Name', 'run',
             'dims', 'modes', 'ISE', 
             'IAE', 'JS', 'KL', 'Entropy'] )
    j=0
    for r in range(repeats):
        experiment = experiment_class(**kwargs)
        #Must be squared since we want a covariance matrix
        if experiment.dimensions == 2 and repeats > 1:
            visualise.plotDensity(experiment.train, "actual", experiment.actualsEstimator.predict, repeats, r, figname=experiment.__name__)
        elif experiment.dimensions == 2 and repeats == 1:
            i=0
            visualise.plotDensity(experiment.train, "actual", experiment.actualsEstimator.predict, len(estimators), i, figname=experiment.__name__)
        
        print("%s %s/%s"%(experiment.__name__, r+1, repeats))       
        
        for name, estimator in estimators.items():            
            print(name)
            start = time.perf_counter()
            
            estimator.dist_ = experiment.dist_loo_
            estimator.fit(experiment.train)
            estimator.dist_ = None            
            
            try:
                models[experiment.__name__]
            except KeyError:
                models[experiment.__name__] = {}
                models[experiment.__name__]['experiment'] = []
            models[experiment.__name__]['experiment'].append(experiment)
            models[experiment.__name__][name] = models[experiment.__name__].get(name, [])+[estimator]

            try:
                results[experiment.__name__]
            except KeyError:
                results[experiment.__name__] = {}
            results[experiment.__name__][name] = results[experiment.__name__].get(name, [])+[array(experiment.getResults(estimator, prekdt=True))]

            if experiment.dimensions == 2 and repeats == 1:
                i+=1
                visualise.plotDensity(experiment.train, name, estimator.predict, len(estimators), i, figname=experiment.__name__)
            
            dfResults.loc[j, ['Experiment', 'Name', 'run', 'dims', 'modes']] = [experiment.__name__, name, r, experiment.dimensions, experiment.modes]
            dfResults.loc[j, ['ISE','IAE', 'JS', 'KL', 'Entropy']] = results[experiment.__name__][name][r]
            j+=1
            
            print('\t\ttime elapsed: %.5f ISE: %.9f'%(time.perf_counter()-start,  dfResults.loc[j-1, ['ISE']][0]))

#    file_path = './data_06_03_2019/%s.csv'%experiment.__name__
#    if update_results:
#        dfResultsMerge = pd.read_csv(file_path, index_col=0)
#        dfResultsMerge.set_index(['Experiment', 'Name', 'run'], inplace=True)
#        dfResults.set_index(['Experiment', 'Name', 'run'], inplace=True)
#        dfResultsMerge.update(dfResults)
#        dfResultsMerge = dfResultsMerge.combine_first(dfResults)
#        dfResultsMerge.reset_index(inplace=True)
#        dfResults.reset_index(inplace=True)
#        dfResultsMerge.to_csv(file_path)
#    else:
#        dfResults.to_csv(file_path)

#resprint = []
#resconsolidate = []
#for experiment_name in results:
#    for kde_name in results[experiment_name]:
#        res = array(results[experiment_name][kde_name])
#        resprint.append(['"%s"'%experiment_name, '"%s"'%kde_name] + ["%f +-%f"%(m,s)  for m, s in zip( res.mean(axis=0), res.std(axis=0) ) ]  )
#        resconsolidate.append( [experiment_name, kde_name] + list(array(list(zip( res.mean(axis=0), res.std(axis=0) ))).flatten())  )
#        
#resprint.sort(key=lambda x: [x[0],x[2]])
#headers = ['"Experiment"', '"Name"', '"IMSE"', '"IMAE"', '"H\'(X)"', '"KL(X\',X)"']
#print( tabulate([headers]+resprint, headers="firstrow") )
#
#dfResults = DataFrame(resconsolidate, columns=['"Experiment"', '"Name"', '"IMSE"', '"std(IMSE)"', '"IMAE"', '"std(IMAE)"', '"H\(X)"', '"std(H\'(X))"', '"KL(X\',X)"', '"std(KL(X\',X))"'])
#dfResults.to_csv('dfResults.csv')
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot(x,y,z)
#ax.scatter(train_set[0], train_set[1], train_set[2])
