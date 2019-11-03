#!/bin/bash
# NOTE: "Should be using conda envt: ecgood"

CURRDIR=$(pwd)
# out=/tmp/analysis/outputs/"$DG_EXPT"_$(date +%Y-%m-%d_%H-%M-%S)
out="${CURRDIR}/analysis/outputs/analysis_$(date +%Y-%m-%d_%H-%M-%S)"
echo "Outputing to ${out}"
echo "Should be using conda envt: ecgood"

# 1) rules based analysis of behavior
DGDIR=/home/lucast4/drawgood/experiments

DG_EXPT=2.3
outthis=$out"_dgrulesmodel_"$DG_EXPT
echo "Assuming already extracted datall_${DG_EXPT}.pickle in drawgood"
echo "1) Rules-based modeling of behavior, running:"
echo "python "$DGDIR"/modelRules.py $DG_EXPT > $outthis"
python "$DGDIR"/modelRules.py $DG_EXPT > $outthis

DG_EXPT=2.2
outthis=$out"_dgrulesmodel_"$DG_EXPT
echo "Assuming already extracted datall_${DG_EXPT}.pickle in drawgood"
echo "1) Rules-based modeling of behavior, running:"
echo "python "$DGDIR"/modelRules.py $DG_EXPT > $outthis"
python "$DGDIR"/modelRules.py $DG_EXPT > $outthis


# 2) get parses
if false; then
    # NOTE: have to enter the correct task names in makeDrawTasks before can run.
    # NOTE, copy S8.2./2 below
    EC_EXPT=S8fixedprim
    outthis=$out"_ecgetparses_"$EC_EXPT
    echo "2) getting parses for ${EC_EXPT}:"
    echo "python analysis/parse.py $EC_EXPT > $outthis"
    python analysis/parse.py $EC_EXPT > $outthis

    EC_EXPT=S9fixedprim
    outthis=$out"_ecgetparses_"$EC_EXPT
    echo "2) getting parses for ${EC_EXPT}:"
    echo "python analysis/parse.py $EC_EXPT > $outthis"
    python analysis/parse.py $EC_EXPT > $outthis
fi

EC_EXPT=S8.2.2
outthis=$out"_ecgetparses_"$EC_EXPT
echo "2) getting parses for ${EC_EXPT}:"
echo "python analysis/parse.py $EC_EXPT > $outthis"
python analysis/parse.py $EC_EXPT > $outthis

EC_EXPT=S9.2
outthis=$out"_ecgetparses_"$EC_EXPT
echo "2) getting parses for ${EC_EXPT}:"
echo "python analysis/parse.py $EC_EXPT > $outthis"
python analysis/parse.py $EC_EXPT > $outthis


# 3) process model parses (--> datflat --> datseg)
EC_EXPT=S8.2.2
outthis=$out"_ecgetdatflatseg_"$EC_EXPT
echo "3) getting datflat/datseg for ${EC_EXPT}:"
echo "python analysis/parse.py $EC_EXPT 0 > $outthis"
python analysis/parse.py $EC_EXPT 0 > $outthis

EC_EXPT=S9.2
outthis=$out"_ecgetdatflatseg_"$EC_EXPT
echo "3) getting datflat/datseg for ${EC_EXPT}:"
echo "python analysis/parse.py $EC_EXPT 0 > $outthis"
python analysis/parse.py $EC_EXPT 0 > $outthis


# 4) get human-model distances
EC_EXPT=S9.2
outthis=$out"_ecmodelhumandists_"$EC_EXPT
echo "4) getting modelHumanDists for ${EC_EXPT}:"
echo "python analysis/getModelHumanDists.py $EC_EXPT > $outthis"
python analysis/getModelHumanDists.py $EC_EXPT > $outthis

EC_EXPT=S8.2.2
outthis=$out"_ecmodelhumandists_"$EC_EXPT
echo "4) getting modelHumanDists for ${EC_EXPT}:"
echo "python analysis/getModelHumanDists.py $EC_EXPT > $outthis"
python analysis/getModelHumanDists.py $EC_EXPT > $outthis


# 5) plot summaries to things
EC_EXPT=S9.2
outthis=$out"_ecsummarize_"$EC_EXPT
echo "5) summarizing ${EC_EXPT}:"
echo "python analysis/summarize.py $EC_EXPT > $outthis"
python analysis/summarize.py $EC_EXPT > $outthis

EC_EXPT=S8.2.2
outthis=$out"_ecsummarize_"$EC_EXPT
echo "5) summarizing ${EC_EXPT}:"
echo "python analysis/summarize.py $EC_EXPT > $outthis"
python analysis/summarize.py $EC_EXPT > $outthis

# === DONE
echo "DONE!"