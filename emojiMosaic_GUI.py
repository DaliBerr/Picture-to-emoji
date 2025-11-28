import os
import tkinter as tk
from tkinter import filedialog, messagebox

# 根据你的主脚本文件名修改这里：
# 假设你的主脚本叫 emojiMosaic.py，并且里面有一个 process(...) 函数
from emojiMosaic import process


def run_gui():
    root = tk.Tk()
    root.title("Emoji Mosaic GUI")
    root.resizable(False, False)

    # ---------------- 路径区域 ----------------
    path_frame = tk.LabelFrame(root, text="路径设置", padx=8, pady=8)
    path_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    dict_var = tk.StringVar()
    emoji_dir_var = tk.StringVar()
    input_var = tk.StringVar()
    out_var = tk.StringVar()

    def choose_dict():
        path = filedialog.askopenfilename(
            title="选择 RGB888 字典文件",
            filetypes=[("Text files", "*.txt *.csv *.dat *.list *.cfg"), ("All files", "*.*")]
        )
        if path:
            dict_var.set(path)

    def choose_emoji_dir():
        path = filedialog.askdirectory(title="选择 emoji 图片目录")
        if path:
            emoji_dir_var.set(path)

    def choose_input():
        path = filedialog.askopenfilename(
            title="选择输入图片",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All files", "*.*")]
        )
        if path:
            input_var.set(path)
            # 自动猜一个输出名
            base, _ = os.path.splitext(path)
            out_var.set(base + "_mosaic.png")

    def choose_out():
        path = filedialog.asksaveasfilename(
            title="选择输出 PNG 路径",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")]
        )
        if path:
            out_var.set(path)

    # 字典
    tk.Label(path_frame, text="RGB888 字典：").grid(row=0, column=0, sticky="e")
    tk.Entry(path_frame, textvariable=dict_var, width=45).grid(row=0, column=1, padx=4)
    tk.Button(path_frame, text="浏览…", command=choose_dict).grid(row=0, column=2, padx=4)

    # emoji 目录
    tk.Label(path_frame, text="Emoji 目录：").grid(row=1, column=0, sticky="e")
    tk.Entry(path_frame, textvariable=emoji_dir_var, width=45).grid(row=1, column=1, padx=4)
    tk.Button(path_frame, text="浏览…", command=choose_emoji_dir).grid(row=1, column=2, padx=4)

    # 输入图
    tk.Label(path_frame, text="输入图片：").grid(row=2, column=0, sticky="e")
    tk.Entry(path_frame, textvariable=input_var, width=45).grid(row=2, column=1, padx=4)
    tk.Button(path_frame, text="浏览…", command=choose_input).grid(row=2, column=2, padx=4)

    # 输出图
    tk.Label(path_frame, text="输出 PNG：").grid(row=3, column=0, sticky="e")
    tk.Entry(path_frame, textvariable=out_var, width=45).grid(row=3, column=1, padx=4)
    tk.Button(path_frame, text="浏览…", command=choose_out).grid(row=3, column=2, padx=4)

    # ---------------- 参数区域 ----------------
    param_frame = tk.LabelFrame(root, text="参数设置", padx=8, pady=8)
    param_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

    # 内部变量
    kernel_var = tk.StringVar(value="box")
    passes_var = tk.IntVar(value=2)
    conv_mode_var = tk.StringVar(value="same")
    block_var = tk.IntVar(value=16)
    step_var = tk.StringVar(value="")  # 空字符串代表 None（=使用 block）
    anchor_var = tk.StringVar(value="center")

    random_factor_var = tk.DoubleVar(value=0.0)
    random_k_var = tk.IntVar(value=10)

    emoji_size_var = tk.IntVar(value=32)
    alpha_th_var = tk.IntVar(value=0)

    # 第一行：kernel / passes / conv_mode
    tk.Label(param_frame, text="卷积核：").grid(row=0, column=0, sticky="e")
    tk.OptionMenu(param_frame, kernel_var, "box", "gauss").grid(row=0, column=1, sticky="w")

    tk.Label(param_frame, text="卷积次数：").grid(row=0, column=2, sticky="e")
    tk.Spinbox(param_frame, from_=0, to=20, textvariable=passes_var, width=5).grid(row=0, column=3, sticky="w")

    tk.Label(param_frame, text="卷积模式：").grid(row=0, column=4, sticky="e")
    tk.OptionMenu(param_frame, conv_mode_var, "same", "valid").grid(row=0, column=5, sticky="w")

    # 第二行：block / step / anchor
    tk.Label(param_frame, text="块大小 block：").grid(row=1, column=0, sticky="e")
    tk.Spinbox(param_frame, from_=2, to=512, textvariable=block_var, width=5).grid(row=1, column=1, sticky="w")

    tk.Label(param_frame, text="步长 step：").grid(row=1, column=2, sticky="e")
    e_step = tk.Entry(param_frame, textvariable=step_var, width=7)
    e_step.grid(row=1, column=3, sticky="w")
    tk.Label(param_frame, text="（留空=使用 block）").grid(row=1, column=4, columnspan=2, sticky="w")

    tk.Label(param_frame, text="锚点：").grid(row=2, column=0, sticky="e")
    tk.OptionMenu(param_frame, anchor_var, "center", "topleft").grid(row=2, column=1, sticky="w")

    # 第三行：随机相关
    tk.Label(param_frame, text="随机因子：").grid(row=3, column=0, sticky="e")
    tk.Scale(
        param_frame,
        from_=0.0, to=1.0,
        resolution=0.05,
        orient=tk.HORIZONTAL,
        variable=random_factor_var,
        length=150
    ).grid(row=3, column=1, columnspan=2, sticky="w")
    tk.Label(param_frame, text="（0=完全最近邻，建议 0~0.3）").grid(row=3, column=3, columnspan=3, sticky="w")

    tk.Label(param_frame, text="随机邻居数 k：").grid(row=4, column=0, sticky="e")
    tk.Spinbox(param_frame, from_=1, to=50, textvariable=random_k_var, width=5).grid(row=4, column=1, sticky="w")
    tk.Label(param_frame, text="（在最近 k 个颜色里随机挑）").grid(row=4, column=2, columnspan=4, sticky="w")

    # 第四行：emoji_size / alpha_th
    tk.Label(param_frame, text="emoji 像素：").grid(row=5, column=0, sticky="e")
    tk.Spinbox(param_frame, from_=8, to=256, textvariable=emoji_size_var, width=5).grid(row=5, column=1, sticky="w")

    tk.Label(param_frame, text="alpha 阈值：").grid(row=5, column=2, sticky="e")
    tk.Scale(
        param_frame,
        from_=0, to=255,
        orient=tk.HORIZONTAL,
        variable=alpha_th_var,
        length=150
    ).grid(row=5, column=3, columnspan=3, sticky="w")

    # ---------------- 执行按钮 ----------------
    btn_frame = tk.Frame(root, padx=8, pady=8)
    btn_frame.grid(row=2, column=0, sticky="e")

    def on_run():
        dict_path = dict_var.get().strip()
        emoji_dir = emoji_dir_var.get().strip()
        input_path = input_var.get().strip()
        out_path = out_var.get().strip()

        if not dict_path or not os.path.isfile(dict_path):
            messagebox.showerror("错误", "请正确选择 RGB888 字典文件")
            return
        if not emoji_dir or not os.path.isdir(emoji_dir):
            messagebox.showerror("错误", "请正确选择 emoji 目录")
            return
        if not input_path or not os.path.isfile(input_path):
            messagebox.showerror("错误", "请正确选择输入图片")
            return
        if not out_path:
            messagebox.showerror("错误", "请设置输出 PNG 路径")
            return

        # 处理 step（允许为空表示 None）
        step_str = step_var.get().strip()
        if step_str == "":
            step_val = None
        else:
            try:
                step_val = int(step_str)
                if step_val <= 0:
                    raise ValueError
            except Exception:
                messagebox.showerror("错误", "step 必须是正整数或留空")
                return

        try:
            # 调用你主脚本里的 process(...)
            process(
                dict_path=dict_path,
                emoji_dir=emoji_dir,
                input_path=input_path,
                out_path=out_path,
                kernel=kernel_var.get(),
                passes=passes_var.get(),
                conv_mode=conv_mode_var.get(),
                block=block_var.get(),
                step=step_val,
                anchor=anchor_var.get(),
                emoji_size=emoji_size_var.get(),
                alpha_threshold=alpha_th_var.get(),
                random_factor=random_factor_var.get(),
                random_k=random_k_var.get()
            )

        except Exception as e:
            messagebox.showerror("运行出错", f"生成马赛克时发生错误：\n{e}")
        else:
            messagebox.showinfo("完成", f"马赛克生成完成！\n已保存到：\n{out_path}")

    tk.Button(btn_frame, text="生成马赛克", command=on_run, width=20).grid(row=0, column=0, padx=5)
    tk.Button(btn_frame, text="退出", command=root.destroy, width=10).grid(row=0, column=1, padx=5)

    root.mainloop()


if __name__ == "__main__":
    run_gui()
