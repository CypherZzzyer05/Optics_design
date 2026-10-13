"""A cached audit is reusable only for the same current physical result files."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from continue_gold_mesh_study import reuse_verified_native_audit


class NativeAuditReuseTests(unittest.TestCase):
    def fixture(self,directory):
        errors={f'nf_{plane}_{vector}_{suffix}':0. for plane in ('mid','top') for vector in ('E','H')
                for suffix in ('enhancement','wavelength_nm','incident_squared','x_nm','y_nm','z_nm')}
        report={'sample_sha256':'sample','project_sha256':'project','status':2,
                'native_geometry_verified':True,'native_spectra_match_export':True,
                'native_fields_match_export':True,'field_monitor_count':4,'native_vs_export_field_errors':errors}
        (directory/'native_audit.json').write_text(json.dumps(report))
        return report

    def test_unchanged_complete_audit_can_be_reused(self):
        with TemporaryDirectory() as temporary:
            directory=Path(temporary)/'Au4';directory.mkdir();self.fixture(directory)
            with patch('continue_gold_mesh_study.validate_saved_case',return_value={'sample_sha256':'sample'}),\
                 patch('continue_gold_mesh_study.checksum',return_value='project'):
                self.assertEqual(reuse_verified_native_audit(directory)['status'],2)

    def test_changed_native_file_or_export_is_rejected(self):
        with TemporaryDirectory() as temporary:
            directory=Path(temporary)/'Au4';directory.mkdir();self.fixture(directory)
            for sample,project in (('sample','changed'),('changed','project')):
                with patch('continue_gold_mesh_study.validate_saved_case',return_value={'sample_sha256':sample}),\
                     patch('continue_gold_mesh_study.checksum',return_value=project):
                    with self.assertRaisesRegex(ValueError,'no longer match'):
                        reuse_verified_native_audit(directory)

    def test_one_missing_native_field_error_prevents_reuse(self):
        with TemporaryDirectory() as temporary:
            directory=Path(temporary)/'Au4';directory.mkdir();report=self.fixture(directory)
            report['native_vs_export_field_errors'].pop('nf_top_H_enhancement')
            (directory/'native_audit.json').write_text(json.dumps(report))
            with patch('continue_gold_mesh_study.validate_saved_case',return_value={'sample_sha256':'sample'}),\
                 patch('continue_gold_mesh_study.checksum',return_value='project'):
                with self.assertRaisesRegex(ValueError,'field audit is incomplete'):
                    reuse_verified_native_audit(directory)


if __name__=='__main__':unittest.main()
