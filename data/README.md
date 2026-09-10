# data/

| 文件 | 来源 | 状态 |
|---|---|---|
| `courses.json` | `~/Downloads/course-catalog.csv`(12,823 行 section 记录 → 4,502 门课),经 `ingest/export_almabot.py` | 待导出(见 issue) |
| `prereqs.json` | 同上,`prereqCodes` 边(~2,800 条),只保留 catalog 内存在的课 | 待导出 |
| `requirements.json` | GenEd 规则来自 `almabot-web/src/lib/gened.ts`;专业 requirement CSV 中没有,待补来源 | 待导出 |
| `chunks.jsonl` | `ingest/chunk.py`,一门课一个 chunk,带 `doc_id` | 待生成 |

注意:`almabot-web/public/courses.json` 里 `desc` 被截到 240 字符、`attrs` 截到 160,且没有学期字段,
**不要直接复制过来**,要从 CSV 重新导出全量文本。

CSV 列:Year, Term, YearTerm, Subject, Number, Name, Description, Credit Hours, Section Info,
Degree Attributes, Schedule Information, CRN, Section, Status Code, Part of Term, Section Title,
Section Credit Hours, Section Status, Enrollment Status, Type, Type Code, Start Time, End Time,
Days of Week, Room, Building, Instructors。当前只有 2026 Spring 一个学期。
