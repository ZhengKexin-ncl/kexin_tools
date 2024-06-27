import mrcfile
import os
import warnings


def get_mrc_files(directory):
    """
    获取指定文件夹中的所有MRC文件
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mrc')]


def get_mrc_header_info(mrc_path):
    """
    获取MRC文件的头部信息
    """
    try:
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            header = mrc.header

            # 收集头部信息
            header_info = []
            header_info.append(f"File: {mrc_path}")
            header_info.append(f"nx (X axis size): {header.nx}")
            header_info.append(f"ny (Y axis size): {header.ny}")
            header_info.append(f"nz (Z axis size): {header.nz}")
            header_info.append(f"mode (Data mode): {header.mode}")
            header_info.append(f"nxstart (Start point of sub-image): {header.nxstart}")
            header_info.append(f"nystart (Start point of sub-image): {header.nystart}")
            header_info.append(f"nzstart (Start point of sub-image): {header.nzstart}")
            header_info.append(f"mx (Grid size in x): {header.mx}")
            header_info.append(f"my (Grid size in y): {header.my}")
            header_info.append(f"mz (Grid size in z): {header.mz}")
            header_info.append(f"cella (Cell dimensions): {header.cella}")
            header_info.append(f"cellb (Cell angles): {header.cellb}")
            header_info.append(f"mapc (Column axis): {header.mapc}")
            header_info.append(f"mapr (Row axis): {header.mapr}")
            header_info.append(f"maps (Section axis): {header.maps}")
            header_info.append(f"dmin (Minimum density value): {header.dmin}")
            header_info.append(f"dmax (Maximum density value): {header.dmax}")
            header_info.append(f"dmean (Mean density value): {header.dmean}")
            header_info.append(f"ispg (Space group number): {header.ispg}")
            header_info.append(f"nsymbt (Bytes used for symmetry data): {header.nsymbt}")
            header_info.append(f"origin (Origin): {header.origin}")

            # 处理 map 字符串
            header_info.append(f"map (Map string): {header.map.tobytes().decode('utf-8', 'ignore')}")
            header_info.append(f"machst (Machine stamp): {header.machst}")
            header_info.append(f"rms (RMS deviation): {header.rms}")
            header_info.append(f"nlabl (Number of labels): {header.nlabl}")
            header_info.append("Labels:")
            for i in range(header.nlabl):
                header_info.append(f"  Label {i + 1}: {header.label[i].decode('utf-8').strip()}")
            header_info.append("\n")

            return "\n".join(header_info)

    except Exception as e:
        warnings.warn(f"Error reading {mrc_path}: {e}", RuntimeWarning)
        return None


def write_headers_to_individual_files(directory):
    """
    将指定文件夹中每个MRC文件的头部信息写入与其同名的文本文件中
    """
    mrc_files = get_mrc_files(directory)
    problematic_files = []
    for mrc_file in mrc_files:
        header_info = get_mrc_header_info(mrc_file)
        if header_info:
            txt_file = mrc_file.replace('.mrc', '.txt')
            with open(txt_file, 'w') as file:
                file.write(header_info)
        else:
            problematic_files.append(mrc_file)

    # 记录无法读取的文件
    if problematic_files:
        with open(os.path.join(directory, 'problematic_files.txt'), 'w') as file:
            file.write("The following files could not be read:\n")
            for pf in problematic_files:
                file.write(f"{pf}\n")


# 使用示例
directory = r'D:\dataset\DeepETPicker_dataset\SampleDatasets\SHREC_2021\raw_data'  # 使用原始字符串字面量处理反斜杠
write_headers_to_individual_files(directory)
print("Headers written to individual text files.")
