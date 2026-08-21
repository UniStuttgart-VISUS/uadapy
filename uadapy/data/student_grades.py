import numpy as np
import scipy.stats as stats
from collections import namedtuple
from uadapy.distributions import DiracDelta, IndependentJoint

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


def student_tom(tol=0.0):
    m1 = DiracDelta(15.0, tol=tol)
    m2 = stats.trapezoid(*fairly_good())
    p1 = stats.norm(loc=14, scale=5.7)
    p2 = stats.uniform(loc=14, scale=2)

    return IndependentJoint([m1, m2, p1, p2])

def student_david(tol=0.0):
    m1 = DiracDelta(9.0, tol=tol)
    m2 = stats.trapezoid(*good())
    p1 = stats.trapezoid(*fairly_good())
    p2 = DiracDelta(10.0, tol=tol)

    return IndependentJoint([m1, m2, p1, p2])

def student_bob(tol=0.0):
    m1 = DiracDelta(6.0, tol=tol)
    m2 = stats.uniform(loc=10, scale=1)
    p1 = stats.uniform(loc=13, scale=7)
    p2 = stats.trapezoid(*good())

    return IndependentJoint([m1, m2, p1, p2])

def student_jane(tol=0.0):
    m1 = stats.trapezoid(*fairly_good())
    m2 = stats.trapezoid(*very_good())
    p1 = DiracDelta(19.0, tol=tol)
    p2 = stats.uniform(loc=10, scale=2)

    return IndependentJoint([m1, m2, p1, p2])

def student_joe(tol=0.0):
    m1 = stats.trapezoid(*very_bad())
    m2 = stats.trapezoid(*fairly_bad())
    p1 = stats.uniform(loc=10, scale=4)
    p2 = DiracDelta(14.0, tol=tol)

    return IndependentJoint([m1, m2, p1, p2])

def student_jack(tol=0.0):
    m1 = DiracDelta(1.0, tol=tol)
    m2 = stats.uniform(loc=4, scale=2)
    p1 = DiracDelta(9.0, tol=tol)
    p2 = stats.uniform(loc=6, scale=3)

    return IndependentJoint([m1, m2, p1, p2])


def students(tol=0.0):
    """
    Return a dictionary of students and their multivariate distributions

    Parameters
    ----------
    tol : float, optional
        The tolerance for the DiracDelta distributions, by default 0.0.
        Each student has at least one subject that is a DiracDelta distribution, 
        which means that the grade for that subject is fixed. 
        The tol parameter allows you to specify a tolerance for these fixed grades, 
        so that they can vary slightly around the fixed value.
        This is especially important if you want to sample the probability density function (PDF)
        of the joint distribution of grades.
    """
    return {
        'Tom': student_tom(tol=tol),
        'David': student_david(tol=tol),
        'Bob': student_bob(tol=tol),
        'Jane': student_jane(tol=tol),
        'Joe': student_joe(tol=tol),
        'Jack': student_jack(tol=tol)
    }



def sample_dataset(n_per_student=1, random_state=None, tol=0.0):
    """
    Sample a dataset of grades for all students and subjects

    Parameters
    ----------
    n_per_student : int, optional
        The number of samples to generate for each student, by default 1
    random_state : int or np.random.RandomState or RNG, optional
        The random state to use for reproducibility, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple (X, y) where X is a (n_per_student * 6, 4) array of grades and y is a (n_per_student * 6,) array of corresponding student names
    """
    X = []
    y = []
    for student_name, d in students(tol=tol).items():
        samples = d.sample(n=n_per_student, random_state=random_state)
        X.append(samples)
        y.append(np.array([student_name]*n_per_student))
    return np.vstack(X), np.concatenate(y)

