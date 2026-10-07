export function pageWindow(total, page = 0, size = 5) {
  const pages = Math.max(1, Math.ceil(total / size));
  page = Math.max(0, Math.min(page, pages - 1));
  return {
    page,
    pages,
    start: page * size,
    end: Math.min(total, (page + 1) * size),
  };
}
