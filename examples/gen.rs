use pyo3_introspection::introspect_cdylib;
use pyo3_introspection::module_stub_files;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 定位编译好的 .so / .pyd / .dylib 文件
    // 确保你已经运行了 `maturin develop` 且开启了 `experimental-inspect`
    let lib_path = Path::new("target/release/lib_faer.so");

    if !lib_path.exists() {
        return Err("找不到库文件，请确认路径正确且已执行 cargo build".into());
    }

    // 2. 执行内省：从 cdylib 中提取元数据模型
    // introspect_cdylib 会解析二进制并返回模型对象
    let introspected_module = introspect_cdylib(lib_path, "_faer")?;

    // 3. 生成 Stub 文件对象
    // module_stub_files 会将模型转换为 Python Stub 结构
    let stub_files = module_stub_files(&introspected_module);

    // 4. 遍历生成的文件并写入磁盘
    // 注意：如果是包结构，可能会有多个文件（如 __init__.pyi）
    for stub_file in stub_files {
        // stub_file.path 通常是 "pyfaer.pyi" 或 "pyfaer/__init__.pyi"
        let path = stub_file.0.as_path();
        let out_path = Path::new(&path);

        // 如果涉及子目录，先创建目录
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(out_path, &stub_file.1)?;
        println!("已生成: {:?}", out_path);
    }

    Ok(())
}
