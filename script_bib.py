from docx import Document

import gspread
from oauth2client.service_account import ServiceAccountCredentials

path = 'bibliofiele-lijst-1968-2018_V6.b-met-EHC-bezit.docx'
document = Document(path)

scope = ['https://spreadsheets.google.com/feeds']
creds = ServiceAccountCredentials.from_json_keyfile_name('ruth-erfgoed.json', scope)
client = gspread.authorize(creds)

sheet = client.open('mappen en affiches').sheet1





paras = document.paragraphs

years = []
lines_years = []

all_text = [a.text.strip() for a in paras]

for i in range(len(all_text)):
    a = all_text[i]  # remove leading ant trailing spaces

    if len(a) == 4:
        # finds all the years!
        years.append(int(a))
        lines_years.append(i)

print(*years)

grid = []
for i in range(len(years)):
    line_start = lines_years[i] + 1     # inclusive
    if i == len(years)-1: # last year
        line_end = len(all_text)
    else:
        line_end = lines_years[i+1]         # exclusive
    lines_one_year = all_text[line_start:line_end]

    lst = [years[i]]
    for j in range(len(lines_one_year)):
        if len(lines_one_year[j]) == 0:
            grid.append(lst)
            lst = [years[i]]
        else:
            lst.append(lines_one_year[j])

    if len(lst) > 1:
        grid.append(lst)

grid_clean = []
# cleaning up the grid:
for i in range(len(grid)):
    grid_i = grid[i]
    print(grid_i)

    assert 1800 < grid_i[0] < 2020, '{}'.format(grid_i[0])

    if len(grid_i) == 1:
        continue    # only year
    elif len(grid_i) == 2 and 'EHC' in grid_i[-1]:
        # add the EHC to previous line
        grid_clean[-1].append(grid_i[-1])

    elif grid_i[0] == 2018:
        for k in range(1, len(grid_i)):
            grid_clean.append([[len(grid_clean) + 1], grid_i[0], grid_i[k]])
        continue

    else:
        # nothing to say about the title
        assert 'Antwerpen' in grid_i[2],  '{}'.format(grid_i[2])

        grid_clean.append([len(grid_clean) + 1] + grid_i)


print(grid_clean)

titles = ['index', 'jaar', 'titel', 'locatie', 'inhoud', 'realisatie', 'afmetingen', 'aantal exemplaren', 'EHC-bezit',
          'waarde']
if 0: # updating:


    # for i in range(len(titles)):
    #     sheet.update_cell(1, 1+i, titles[i])

    sheet.insert_row(titles, 1)

    # for i in range(len(years)):
    #     sheet.update_cell(2 + i, 1, years[i])

    # for i in range(len(grid_clean)):
    #     for j in range(len(grid_clean[i])):
    #         sheet.update_cell(2 + i, j+1, grid_clean[i][j])

    # for i in range(len(grid_clean)):
    #     sheet.insert_row(grid_clean[i], 2 + i)

    # Select a range
    cell_list = sheet.range('A2:M{}'.format(len(grid_clean)+1))

    for i in range(len(grid_clean)):
        for j in range(len(grid_clean[i])):
            cell_list[i][j] = grid_clean[i][j]

    # for cell in cell_list:
    #     cell.value = 'O_o'

    # Update in batch
    sheet.update_cells(cell_list)

else:
    # save directly as xlsx

    import xlwt

    book = xlwt.Workbook() #encoding="utf-8")

    sheet1 = book.add_sheet("Sheet 1")


    for j in range(len(titles)):
        sheet1.write(0, j,  titles[j])

    for i in range(len(grid_clean)):
        for j in range(len(grid_clean[i])):
            sheet1.write(i+1, j,  str(grid_clean[i][j]))

    book.save('lijst_affiches_v2.xls')
