#!/usr/bin/env python
"""Score diarization system output.

To evaluate system output stored in RTTM files ``sys1.rttm``, ``sys2.rttm``,
... against a corresponding reference diarization stored in RTTM files
``ref1.rttm``, ``ref2.rttm``, ...:

    python score.py -r ref1.rttm ref2.rttm ... -s sys1.rttm sys2.rttm ...

which will calculate and report the following metrics both overall and on
a per-file basis:

- diarization error rate (DER)
- jaccard error rate (JER)
- B-cubed precision (B3-Precision)
- B-cubed recall (B3-Recall)
- B-cubed F1 (B3-F1)
- Goodman-Kruskal tau in the direction of the reference diarization to the
  system diarization (GKT(ref, sys))
- Goodman-Kruskal tau in the direction of the system diarization to the
  reference diarization (GKT(sys, ref))
- conditional entropy of the reference diarization given the system
  diarization in bits (H(ref|sys))
- conditional entropy of the system diarization given the reference
  diarization in bits (H(sys|ref))
- mutual information in bits (MI)
- normalized mutual information (NMI)

Alternately, we could have specified the reference and system RTTM files via
script files of paths (one per line) using the ``-R`` and ``-S`` flags:

    python score.py -R ref.scp -S sys.scp

By default the scoring regions for each file will be determined automatically
from the reference and speaker turns. However, it is possible to specify
explicit scoring regions using a NIST un-partitioned evaluation map (UEM) file
and the ``-u`` flag. For instance, the following:

    python score.py -u all.uem -R ref.scp -S sys.scp

will load the files to be scored + scoring regions from ``all.uem``, filter out
and warn about any speaker turns not present in those files, and trim the
remaining turns to the relevant scoring regions before computing the metrics
as before.

Diarization error rate (DER) is scored using the NIST ``md-eval.pl`` tool with
a default collar size of 0 ms and explicitly including regions that contain
overlapping speech in the reference diarization. If desired, this behavior
can be altered using the ``--collar`` and ``--ignore_overlaps`` flags. For
instance

    python score.py --collar 0.100 --ignore_overlaps -R ref.scp -S sys.scp

would compute DER using a 100 ms collar and with overlapped speech ignored.
All other metrics are computed off of frame-level labelings generated from the
reference and system speaker turns **WITHOUT** any use of collars. The default
frame step is 10 ms, which may be altered via the ``--step`` flag. For more
details, consult the docstrings within the ``scorelib.metrics`` module.

The overall and per-file results will be printed to STDOUT as a table formatted
using the ``tabulate`` package. Some basic control of the formatting of this
table is possible via the ``--n_digits`` and ``--table_format`` flags. The
former controls the number of decimal places printed for floating point
numbers, while the latter controls the table format. For a list of valid
table formats plus example outputs, consult the documentation for the
``tabulate`` package:

    https://pypi.python.org/pypi/tabulate
"""
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import glob

from tabulate import tabulate

from .scorelib import __version__ as VERSION
from .scorelib.argparse import ArgumentParser
from .scorelib.rttm import load_rttm
from .scorelib.turn import merge_turns, trim_turns
from .scorelib.score import score
from .scorelib.six import iterkeys
from .scorelib.uem import gen_uem, load_uem
from .scorelib.utils import error, info, warn, xor
from types import SimpleNamespace


def load_rttms(rttm_fns):
    """Load speaker turns from RTTM files.

    Parameters
    ----------
    rttm_fns : list of str
        Paths to RTTM files.

    Returns
    -------
    turns : list of Turn
        Speaker turns.

    file_ids : set
        File ids found in ``rttm_fns``.
    """
    turns = []
    file_ids = set()
    for rttm_fn in rttm_fns:
        if not os.path.exists(rttm_fn):
            error('Unable to open RTTM file: %s' % rttm_fn)
            sys.exit(1)
        try:
            turns_, _, file_ids_ = load_rttm(rttm_fn)
            turns.extend(turns_)
            file_ids.update(file_ids_)
        except IOError as e:
            error('Invalid RTTM file: %s. %s' % (rttm_fn, e))
            sys.exit(1)
    return turns, file_ids


def check_for_empty_files(ref_turns, sys_turns, uem):
    """Warn on files in UEM without reference or speaker turns."""
    ref_file_ids = {turn.file_id for turn in ref_turns}
    sys_file_ids = {turn.file_id for turn in sys_turns}
    # for file_id in sorted(iterkeys(uem)):
    #     if file_id not in ref_file_ids:
    #         warn('File "%s" missing in reference RTTMs.' % file_id)
    #     if file_id not in sys_file_ids:
    #         warn('File "%s" missing in system RTTMs.' % file_id)
    # TODO: Clarify below warnings; this indicates that there are no
    #       ELIGIBLE reference/system turns.
    if not ref_turns:
        warn('No reference speaker turns found within UEM scoring regions.')
    if not sys_turns:
        warn('No system speaker turns found within UEM scoring regions.')
    return set.intersection(ref_file_ids, sys_file_ids)


def load_script_file(fn):
    """Load file names from ``fn``."""
    with open(fn, 'rb') as f:
        return [line.decode('utf-8').strip() for line in f]


def print_table(file_scores, global_scores, n_digits=2,
                table_format='simple'):
    """Pretty print scores as table.

    Parameters
    ----------
    file_to_scores : dict
        Mapping from file ids in ``uem`` to ``Scores`` instances.

    global_scores : Scores
        Global scores.

    n_digits : int, optional
        Number of decimal digits to display.
        (Default: 3)

    table_format : str, optional
        Table format. Passed to ``tabulate.tabulate``.
        (Default: 'simple')
    """
    col_names = ['File',
                 'DER', # Diarization error rate.
                 'JER', # Jaccard error rate.
                 'B3-Precision', # B-cubed precision.
                 'B3-Recall', # B-cubed recall.
                 'B3-F1', # B-cubed F1.
                 'GKT(ref, sys)', # Goodman-Krustal tau (ref, sys).
                 'GKT(sys, ref)', # Goodman-Kruskal tau (sys, ref).
                 'H(ref|sys)',  # Conditional entropy of ref given sys.
                 'H(sys|ref)',  # Conditional entropy of sys given ref.
                 'MI', # Mutual information.
                 'NMI', # Normalized mutual information.
                ]
    rows = sorted(file_scores, key=lambda x: x.file_id)
    rows.append(global_scores._replace(file_id='*** OVERALL ***'))
    floatfmt = '.%df' % n_digits
    tbl = tabulate(
        rows, headers=col_names, floatfmt=floatfmt, tablefmt=table_format)
    print(tbl)


class ScoreKeeper:

    def __init__(self, **kwargs):
        self.args = {
            'ref_rttm_fns': [],
            'ref_rttm_scpf': None,
            'sys_rttm_fns': [],
            'sys_rttm_scpf': None,
            'uemf': None,
            'collar': 0.0,
            'ignore_overlaps': False,
            'jer_min_ref_dur': 0.0,
            'step': 0.010,
            'n_digits': 2,
            'table_format': 'simple'}
        self.args.update(kwargs)
        self.args = SimpleNamespace(**self.args)

        if not xor(self.args.ref_rttm_fns, self.args.ref_rttm_scpf):
            error('Exactly one of ref_rttm_fns and ref_rttm_scpf must be set.')

        # Check that at least one reference RTTM and at least one system RTTM
        # was specified.
        if self.args.ref_rttm_scpf is not None:
            self.args.ref_rttm_fns = load_script_file(self.args.ref_rttm_scpf)
        if not self.args.ref_rttm_fns:
            error('No reference RTTMs specified.')
            sys.exit(1)

        # Load speaker/reference speaker turns and UEM. If no UEM specified,
        # determine it automatically.
        print('Loading speaker turns from reference RTTMs...',)
        self.ref_turns, _ = load_rttms(self.args.ref_rttm_fns)

        if self.args.uemf is not None:
            print('Loading universal evaluation map...')
            self.uem = load_uem(self.args.uemf)
        else:
            warn('No universal evaluation map specified. Approximating from '
                 'reference and speaker turn extents...')
            self.uem = gen_uem(self.ref_turns, self.sys_turns)

        # Trim turns to UEM scoring regions and merge any that overlap.
        print('Trimming reference speaker turns to UEM scoring regions...')
        self.ref_turns = trim_turns(self.ref_turns, self.uem)
        print('Checking for overlapping reference speaker turns...')
        self.ref_turns = merge_turns(self.ref_turns)

    def score(self, sys_rttm_fns=None, sys_rttm_scpf=None):
        if not xor(sys_rttm_fns, sys_rttm_scpf):
            error('Exactly one of sys_rttm_fns and sys_rttm_scpf must be set.')

        if sys_rttm_scpf is not None:
            sys_rttm_fns = load_script_file(sys_rttm_scpf)

        if not sys_rttm_fns:
            error('No system RTTMs specified.')
            sys.exit(1)

        print('Loading speaker turns from system RTTMs...')
        sys_turns, _ = load_rttms(sys_rttm_fns)

        print('Trimming system speaker turns to UEM scoring regions...')
        sys_turns = trim_turns(sys_turns, self.uem)

        print('Checking for overlapping system speaker turns...')
        sys_turns = merge_turns(sys_turns)

        print('Scoring...')
        subset = check_for_empty_files(self.ref_turns, sys_turns, self.uem)
        subset_ref_turns = [t for t in self.ref_turns if t.file_id in subset]
        subset_uem = {k: v for k, v in self.uem.items() if k in subset}
        file_scores, global_scores = score(
            subset_ref_turns, sys_turns, subset_uem, step=self.args.step,
            jer_min_ref_dur=self.args.jer_min_ref_dur, collar=self.args.collar,
            ignore_overlaps=self.args.ignore_overlaps)
        print_table(file_scores, global_scores, self.args.n_digits,
            self.args.table_format)
        file_scores = [fs._asdict() for fs in file_scores]
        global_scores = global_scores._asdict()
        global_scores.pop('file_id')
        return file_scores, global_scores


if __name__ == '__main__':
    ref_rttm_dir = '../data/test/rttm/'
    sys_rttm = '../rttm/sample.rttm'
    uemf = '../data/test/uem/all.uem'
    ref_rttms = [f for f in glob.glob('../data/test/rttm/*.rttm')]
    sk = ScoreKeeper(
        ref_rttm_fns=ref_rttms,
        uemf=uemf)
    print(sk.score(sys_rttm_fns=[sys_rttm]))
