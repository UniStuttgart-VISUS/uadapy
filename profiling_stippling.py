'''
profiling execution times.
Maybe need to run twice because first time numba will recompile 
(due to relevant files having been changed) and therfore distort timings.
Ideally you run the code you want to profile and afterwards profile it.
'''
import single_dist_stippling
fig,_ = single_dist_stippling.main(steps=10)

import cProfile
cProfile.run('single_dist_stippling.main(steps=1000)', 'profilingresult.txt', 'cumtime')

import pstats
p = pstats.Stats('profilingresult.txt')
p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)