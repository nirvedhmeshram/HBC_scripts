from iree.compiler import tf as tfc

imported_ir = tfc.compile_saved_model("hbc_sm_v2_largesize/", import_only=True, save_temp_mid_level_input="mhlo_hbc_sm_v2_large.mlir")
