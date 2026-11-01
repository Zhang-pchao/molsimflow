"""Deterministic live-tail and periodic-junction regression checks."""
import io,shutil,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
from molsimflow.io.live_snapshot import copy_prefix,extract_prefix_snapshot,last_complete_stream
from molsimflow.postprocess.junction_species import (
    zone_from_x,periodic_translation,changed_edges,regional_edge_changes,region_species_counts)

def frame(step=0):
    return ("ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 20\n0 20\n0 30\n"
            "ITEM: ATOMS id type x y z\n1 8 0.2 2 3\n2 2 19 2 3\n")%step

class SnapshotTests(unittest.TestCase):
    def test_last_complete_before_partial_tail(self):
        result,meta=last_complete_stream(io.StringIO(frame()+frame(10)+frame(20)[:-8]),2)
        self.assertEqual(result,frame(10));self.assertEqual(meta["complete_frames_seen"],2)
        self.assertTrue(meta["incomplete_text_tail_ignored"])
    def test_count_corruption_rejected(self):
        with self.assertRaisesRegex(ValueError,"atom count"):
            last_complete_stream(io.StringIO(frame().replace("ATOMS\n2","ATOMS\n3")),2)
    def test_interior_corruption_rejected(self):
        with self.assertRaisesRegex(ValueError,"column count"):
            last_complete_stream(io.StringIO(frame().replace("1 8 0.2 2 3","1 8 0.2 2")),2)
    def test_empty_input_rejected(self):
        with self.assertRaisesRegex(ValueError,"No complete"):
            last_complete_stream(io.StringIO("ITEM: TIMESTEP\n"),2)
    def test_capture_is_bounded_when_source_grows(self):
        with tempfile.TemporaryDirectory() as td:
            a,b,out=[Path(td)/x for x in ["source","prefix","snapshot"]]
            a.write_text(frame());meta=copy_prefix(a,b)
            a.write_text(frame()+frame(10))
            result=extract_prefix_snapshot(b,out,2)
            self.assertEqual(meta["source_prefix_bytes"],len(frame()))
            self.assertEqual(result["last_frame_timestep"],0)
    @unittest.skipUnless(shutil.which("zstd"),"zstd executable unavailable")
    def test_truncated_zstd_tail_keeps_complete_frame(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"prefix.zst";out=Path(td)/"snapshot"
            data=subprocess.check_output(["zstd","-q","-c"],input=frame().encode())
            p.write_bytes(data[:-2])
            meta=extract_prefix_snapshot(p,out,2)
            self.assertEqual(out.read_text(),frame())
            self.assertTrue(meta["incomplete_compressed_tail_ignored"])
    @unittest.skipUnless(shutil.which("zstd"),"zstd executable unavailable")
    def test_corrupt_zstd_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"prefix.zst";p.write_bytes(b"this is not a zstd stream")
            with self.assertRaises(ValueError):
                extract_prefix_snapshot(p,Path(td)/"out",2)

class JunctionTests(unittest.TestCase):
    def test_both_periodic_and_internal_seams(self):
        self.assertEqual(zone_from_x(-.1,100,6),"periodic_seam")
        self.assertEqual(zone_from_x(101.,100,6),"periodic_seam")
        self.assertEqual(zone_from_x(49.,100,6),"internal_seam")
        self.assertEqual(zone_from_x(25.,100,6),"left_interior")
        self.assertEqual(zone_from_x(75.,100,6),"right_interior")
    def test_global_translation_across_periodic_edge(self):
        ref=np.array([[99.,2,3],[20.,3,4]])
        cur=np.array([[1.,2.2,3],[22.,3.2,4]])
        np.testing.assert_allclose(periodic_translation(ref,cur,[100,100,100]),[2,.2,0],atol=1e-12)
    def test_atom_identity_edges_not_just_coordination(self):
        change=changed_edges([(1,5),(2,6)],[(1,6),(2,5)])
        self.assertEqual(len(change["lost"]),2);self.assertEqual(len(change["added"]),2)
        regions=regional_edge_changes(change,{1:"internal_seam",2:"periodic_seam"})
        self.assertEqual(regions["internal_seam"],{"lost":1,"added":1})
    def test_counts_are_conserved_between_regions(self):
        counts=region_species_counts(["H2O","H2O","N2"],["left_interior","internal_seam","periodic_seam"])
        self.assertEqual(sum(x.get("H2O",0) for x in counts.values()),2)
if __name__=="__main__":unittest.main()
