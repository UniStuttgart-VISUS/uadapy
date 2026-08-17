import numpy as np
import scipy.stats as stats
from collections import namedtuple

def _trapezoid_abcd2cdls(a,b,c,d):
    """
    convert trapezoidal parameters a,b,c,d to c',d',location, scale as expected by scipy.stats.trapezoid
    a = loc
    d = loc + scale
    b = loc + c'*scale
    c = loc + d'*scale
    """
    loc = a
    scale = d - a
    c_ = (b-a)/scale
    d_ = (c-a)/scale
    params = namedtuple('TrapezoidCDLS', ['c', 'd', 'loc', 'scale'])
    return params(c_,d_,loc,scale)

# define trapezoid parameters for textual grades
def very_bad():
    return _trapezoid_abcd2cdls(0, 0, 2, 6)

def bad():
    return _trapezoid_abcd2cdls(2, 4, 6, 7)

def fairly_bad():
    return _trapezoid_abcd2cdls(5, 7, 9, 10)

def fairly_good():
    return _trapezoid_abcd2cdls(10, 11, 13, 14)

def good():
    return _trapezoid_abcd2cdls(13, 14, 16, 18)

def very_good():
    return _trapezoid_abcd2cdls(14, 18, 20, 20)


# define the students and subjects
def subject_names():
    return ['M1', 'M2', 'P1', 'P2']


def distrib_function_names():
    return ['samplers', 'densities', 'means', 'variances']

samplers = namedtuple('samplers', subject_names())
densitites = namedtuple('densities', subject_names())
means = namedtuple('means', subject_names())
variances = namedtuple('variances', subject_names())
distrib_functions = namedtuple('distrib_functions', distrib_function_names())

def student_tom():


    samplers = namedtuple('samplers', subject_names())
    s = samplers(
        lambda n, rand_state=None: np.ones(n)*15, # M1
        lambda n, rand_state=None: stats.trapezoid.rvs(*fairly_good(),size=n, random_state=rand_state), # M2
        lambda n, rand_state=None: stats.norm.rvs(loc=14, scale=5.7, size=n, random_state=rand_state), # P1
        lambda n, rand_state=None: stats.uniform.rvs(loc=14, scale=2, size=n, random_state=rand_state)  # P2 [14,16]
    )
    densitites = namedtuple('densities', subject_names())
    d = densitites(
        lambda x, delta=0: np.where((x>=15-delta) & (x<=15+delta), 1/(2*delta), 0), # M1
        lambda x, delta=0: stats.trapezoid.pdf(x, *fairly_good()), # M2
        lambda x, delta=0: stats.norm.pdf(x, loc=14, scale=5.7), # P1
        lambda x, delta=0: stats.uniform.pdf(x, loc=14, scale=2)  # P2 [14,16]
    ) 
    return s,d

def student_david():
    samplers = namedtuple('samplers', subject_names())
    s = samplers(
        lambda n, rand_state=None: np.ones(n)*9, # M1
        lambda n, rand_state=None: stats.trapezoid.rvs(*good(),size=n, random_state=rand_state), # M2
        lambda n, rand_state=None: stats.trapezoid.rvs(*fairly_good(),size=n, random_state=rand_state), # P1
        lambda n, rand_state=None: np.ones(n)*10  # P2
    )
    densitites = namedtuple('densities', subject_names())
    d = densitites(
        lambda x, delta=0: np.where((x>=9-delta) & (x<=9+delta), 1/(2*delta), 0), # M1
        lambda x, delta=0: stats.trapezoid.pdf(x, *good()), # M2
        lambda x, delta=0: stats.trapezoid.pdf(x, *fairly_good()), # P1
        lambda x, delta=0: np.where((x>=10-delta) & (x<=10+delta), 1/(2*delta), 0)  # P2
    )
    return s,d

def student_bob():
    samplers = namedtuple('samplers', subject_names())
    s = samplers(
        lambda n, rand_state=None: np.ones(n)*6, # M1
        lambda n, rand_state=None: stats.uniform.rvs(loc=10, scale=1, size=n, random_state=rand_state),
        lambda n, rand_state=None: stats.uniform.rvs(loc=13, scale=7, size=n, random_state=rand_state),
        lambda n, rand_state=None: stats.trapezoid.rvs(*good(),size=n, random_state=rand_state)
    )
    densitites = namedtuple('densities', subject_names())
    d = densitites(
        lambda x, delta=0: np.where((x>=6-delta) & (x<=6+delta), 1/(2*delta), 0), # M1
        lambda x, delta=0: stats.uniform.pdf(x, loc=10, scale=1), # M2
        lambda x, delta=0: stats.uniform.pdf(x, loc=13, scale=7), # P1
        lambda x, delta=0: stats.trapezoid.pdf(x, *good())  # P2
    )
    return s,d

def student_jane():
    samplers = namedtuple('samplers', subject_names())
    s = samplers(
        lambda n, rand_state=None: stats.trapezoid.rvs(*fairly_good(),size=n, random_state=rand_state),
        lambda n, rand_state=None: stats.trapezoid.rvs(*very_good(),size=n, random_state=rand_state),
        lambda n, rand_state=None: np.ones(n)*19,
        lambda n, rand_state=None: stats.uniform.rvs(loc=10, scale=2, size=n, random_state=rand_state)
    )
    densitites = namedtuple('densities', subject_names())
    d = densitites(
        lambda x, delta=0: stats.trapezoid.pdf(x, *fairly_good()), # M1
        lambda x, delta=0: stats.trapezoid.pdf(x, *very_good()), # M2
        lambda x, delta=0: np.where((x>=19-delta) & (x<=19+delta), 1/(2*delta), 0), # P1
        lambda x, delta=0: stats.uniform.pdf(x, loc=10, scale=2)  # P2
    )
    return s,d

def student_joe():
    samplers = namedtuple('samplers', subject_names())
    s = samplers(
        lambda n, rand_state=None: stats.trapezoid.rvs(*very_bad(),size=n, random_state=rand_state),
        lambda n, rand_state=None: stats.trapezoid.rvs(*fairly_bad(),size=n, random_state=rand_state),
        lambda n, rand_state=None: stats.uniform.rvs(loc=10, scale=4, size=n, random_state=rand_state),
        lambda n, rand_state=None: np.ones(n)*14
    )
    densitites = namedtuple('densities', subject_names())
    d = densitites(
        lambda x, delta=0: stats.trapezoid.pdf(x, *very_bad()), # M1
        lambda x, delta=0: stats.trapezoid.pdf(x, *fairly_bad()), # M2
        lambda x, delta=0: stats.uniform.pdf(x, loc=10, scale=4), # P1
        lambda x, delta=0: np.where((x>=14-delta) & (x<=14+delta), 1/(2*delta), 0)  # P2
    )
    return s,d

def student_jack():
    samplers = namedtuple('samplers', subject_names())
    s = samplers(
        lambda n, rand_state=None: np.ones(n)*1,
        lambda n, rand_state=None: stats.uniform.rvs(loc=4, scale=2, size=n, random_state=rand_state),
        lambda n, rand_state=None: np.ones(n)*9,
        lambda n, rand_state=None: stats.uniform.rvs(loc=6, scale=3, size=n, random_state=rand_state)
    )
    densitites = namedtuple('densities', subject_names())
    d = densitites(
        lambda x, delta=0: np.where((x>=1-delta) & (x<=1+delta), 1/(2*delta), 0), # M1
        lambda x, delta=0: stats.uniform.pdf(x, loc=4, scale=2), # M2
        lambda x, delta=0: np.where((x>=9-delta) & (x<=9+delta), 1/(2*delta), 0), # P1
        lambda x, delta=0: stats.uniform.pdf(x, loc=6, scale=3)  # P2
    )
    return s,d


def students():
    """
    Return a dictionary of students and their (samplers,pdfs) for each subject
    """
    return {
        'Tom': student_tom(),
        'David': student_david(),
        'Bob': student_bob(),
        'Jane': student_jane(),
        'Joe': student_joe(),
        'Jack': student_jack()
    }

def get_student_samplers():
    """
    Return a dictionary of students and their samplers for each subject
    """
    return {name: sampler_pdf[0] for name, sampler_pdf in students().items()}

def get_student_pdfs():
    """
    Return a dictionary of students and their pdfs for each subject
    """
    return {name: sampler_pdf[1] for name, sampler_pdf in students().items()}


def multivariate_sample(student_samplers, n=1, random_state=None):
    """
    Given a student's samplers namedtuple, sample n grades for each subject and return a (n,4) array

    Parameters
    ----------
    student_samplers : namedtuple
        A namedtuple containing the samplers for each subject for a student
    n : int, optional
        The number of samples to generate for each subject, by default 1
    random_state : int or np.random.RandomState, optional
        The random state to use for reproducibility, by default None
    """
    M1,M2,P1,P2 = student_samplers
    return np.array([
        M1(n, random_state),
        M2(n, random_state),
        P1(n, random_state),
        P2(n, random_state)
    ]).T


def sample_dataset(n_per_student=1, random_state=None):
    """
    Sample a dataset of grades for all students and subjects

    Parameters
    ----------
    n_per_student : int, optional
        The number of samples to generate for each student, by default 1
    random_state : int or np.random.RandomState, optional
        The random state to use for reproducibility, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple (X, y) where X is a (n_per_student * 6, 4) array of grades and y is a (n_per_student * 6,) array of corresponding student names
    """
    X = []
    y = []
    for student_name, samplers in get_student_samplers().items():
        samples = multivariate_sample(samplers, n=n_per_student, random_state=random_state)
        X.append(samples)
        y.append(np.array([student_name]*n_per_student))
    return np.vstack(X), np.concatenate(y)

