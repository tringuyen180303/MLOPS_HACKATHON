import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_metrics
from evidently.test_preset import DataStabilityPreset, NoTargetPerformanceTestPreset, RegressionTestPreset


data_columns = ColumnMapping()
#data_drift_report = Report(metrics=[DataDriftPreset()])

report = Report(metrics=[
    DataDriftPreset(), 
])

report.run(reference_data=reference, current_data=current)
report


report = Report(metrics=[
    ColumnSummaryMetric(column_name='AveRooms'),
    ColumnQuantileMetric(column_name='AveRooms', quantile=0.25),
    ColumnDriftMetric(column_name='AveRooms'),
    
])

report.run(reference_data=reference, current_data=current)
report



tests = TestSuite(tests=[
    TestNumberOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestNumberOfDriftedColumns(),
])

tests.run(reference_data=reference, current_data=current)
tests


suite = TestSuite(tests=[
    NoTargetPerformanceTestPreset(),
])

suite.run(reference_data=reference, current_data=current)
suite


suite = TestSuite(tests=[
    TestNumberOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestNumberOfDriftedColumns(),
    TestColumnDrift('Population'),
    TestShareOfOutRangeValues('Population'),
    DataStabilityTestPreset(),
    RegressionTestPreset()
    
])

suite.run(reference_data=reference, current_data=current)