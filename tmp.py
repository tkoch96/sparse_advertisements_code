rows = []
for row in open('screenlog.0','r'):
	if "Current cache size" in row:
		continue
	rows.append(row)
with open('newscreenlog.0','w') as f:
	for row in rows:
		f.write(row)