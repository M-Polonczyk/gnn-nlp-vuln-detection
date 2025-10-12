import json
from pathlib import Path

# graph_dir = Path('../data/preprocessed/megavul/c_cpp/2023/graph')

item9 = {
    "cve_id": "CVE-2017-13011",
    "cwe_ids": ["CWE-119"],
    # "cvss_vector": "AV:N/AC:L/Au:N/C:P/I:P/A:P",
    # "cvss_is_v3": False,
    # "repo_name": "the-tcpdump-group/tcpdump",
    # "commit_msg": "CVE-2017-13011/Properly check for buffer overflow in bittok2str_internal().\n\nAlso, make the buffer bigger.\n\nThis fixes a buffer overflow discovered by Bhargava Shastry,\nSecT/TU Berlin.\n\nAdd a test using the capture file supplied by the reporter(s), modified\nso the capture file won't be rejected as an invalid capture.",
    "commit_hash": "9f0730bee3eb65d07b49fd468bc2f269173352fe",
    "git_url": "https://github.com/the-tcpdump-group/tcpdump/commit/9f0730bee3eb65d07b49fd468bc2f269173352fe",
    "file_path": "util-print.c",
    "func_name": "unsigned_relts_print",
    # "func_before": None,
    # "abstract_func_before": None,
    # "func_graph_path_before": None,
    "func": 'void\nunsigned_relts_print(netdissect_options *ndo,\n                     uint32_t secs)\n{\n\tstatic const char *lengths[] = {"y", "w", "d", "h", "m", "s"};\n\tstatic const u_int seconds[] = {31536000, 604800, 86400, 3600, 60, 1};\n\tconst char **l = lengths;\n\tconst u_int *s = seconds;\n\n\tif (secs == 0) {\n\t\tND_PRINT((ndo, "0s"));\n\t\treturn;\n\t}\n\twhile (secs > 0) {\n\t\tif (secs >= *s) {\n\t\t\tND_PRINT((ndo, "%d%s", secs / *s, *l));\n\t\t\tsecs -= (secs / *s) * *s;\n\t\t}\n\t\ts++;\n\t\tl++;\n\t}\n}',
    # "abstract_func": 'void\nunsigned_relts_print(netdissect_options *VAR_0,\n                     uint32_t VAR_1)\n{\n\tstatic const char *VAR_2[] = {"y", "w", "d", "h", "m", "s"};\n\tstatic const u_int VAR_3[] = {31536000, 604800, 86400, 3600, 60, 1};\n\tconst char **VAR_4 = VAR_2;\n\tconst u_int *VAR_5 = VAR_3;\n\n\tif (VAR_1 == 0) {\n\t\tND_PRINT((VAR_0, "0s"));\n\t\treturn;\n\t}\n\twhile (VAR_1 > 0) {\n\t\tif (VAR_1 >= *VAR_5) {\n\t\t\tND_PRINT((VAR_0, "%d%s", VAR_1 / *VAR_5, *VAR_4));\n\t\t\tVAR_1 -= (VAR_1 / *VAR_5) * *VAR_5;\n\t\t}\n\t\tVAR_5++;\n\t\tVAR_4++;\n\t}\n}',
    # "func_graph_path": "the-tcpdump-group/tcpdump/9f0730bee3eb65d07b49fd468bc2f269173352fe/util-print.c/non_vul/1.json",
    # "diff_func": None,
    # "diff_line_info": None,
    "is_vul": False, # label
}

with Path("data/preprocessed/megavul/c_cpp/2023/megavul_simple.json").open(mode="r") as f:
    megavul = json.load(f)
    item = megavul[9]
    cve_id = item["cve_id"] # CVE-2022-24786
    # cvss_vector = item['cvss_vector']   # AV:N/AC:L/Au:N/C:P/I:P/A:P
    is_vul = item["is_vul"] # True
    if is_vul:
        func_before = item["func_before"]  # vulnerable function

    func_after = item["func"]   # after vul function fixed(i.e., clean function)
    abstract_func_after = item["abstract_func"]

    diff_line_info = item["diff_line_info"] # {'deleted_lines': ['pjmedia_rtcp_comm .... ] , 'added_lines': [ .... ] }
    git_url = item["git_url"]   # https://github.com/pjsip/pjproject/commit/11559e49e65bdf00922ad5ae28913ec6a198d508

    # if item['func_graph_path_before'] is not None: # graphs of some functions cannot be exported successfully
    #     graph_file_path = graph_dir / item['func_graph_path_before']
    #     graph_file = json.load(graph_file_path.open(mode='r'))
    #     nodes, edges = graph_file['nodes'] , graph_file['edges']
    #     print(nodes)    # [{'version': '0.1', 'language': 'NEWC', '_label': 'META_DATA', 'overlays': ....
    #     print(edges)    # [{'innode': 196, 'outnode': 2, 'etype': 'AST', 'variable': None}, ...]

    print(f"Item: {item}\n\n")
    print(f"cve_id: {cve_id}, cvss_vector: {cvss_vector}, is_vul: {is_vul}")
