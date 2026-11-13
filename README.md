# PATCH

Un cortometraje animado de 60 segundos hecho **a puro código**: el mundo 3D, los personajes y sus esqueletos, la animación, las cámaras, la luz, la música, los efectos de sonido y el montaje salen de este repositorio. No hay grabaciones, imágenes generadas ni programas de animación.

> Un ranger solitario y su viejo perro tuerto, Patch, rescatan a un cachorro de un convoy de cazadores enmascarados en un cañón del desierto.

Hecho por [@promptmedia](https://instagram.com/promptmedia) con Claude.

## Cómo funciona

| Archivo | Qué hace |
|---|---|
| `engine.py` | Motor de render toon en OpenGL: sombreado plano en tres tonos, rim light duro, mapa de sombras, contornos gruesos, niebla violeta y cielo de atardecer. |
| `geo.py` | Primitivas (cajas, cilindros, esferas) y matrices de cámara. |
| `rig.py` | Esqueletos y personajes: el ranger, los cazadores enmascarados, el jefe, Patch, el cachorro, las camionetas y la jaula. También define las poses y los ciclos de carrera, caminata y galope. |
| `world.py` | El cañón (heightfield con estratos), las rocas, los cactus, las mesetas y los efectos: polvo, chispas, fogonazos y explosión. |
| `film.py` | La película: 18 planos con su animación, cámara, subtítulos y marcas de sonido. |
| `audio.py` | Banda sonora sintetizada: guitarra (Karplus-Strong), silbido, taikos, bajo, disparos con eco de cañón, motores, choque, explosión, gruñidos y gemidos. |
| `render.py` | Renderiza hojas de cuadros de prueba o los planos finales en MP4. |
| `assemble.py` | Une los planos, agrega el audio, genera el `.srt` y la hoja de cuadros. |

## Requisitos

- Python 3.11 o superior
- ffmpeg en el PATH
- Un contexto OpenGL 3.3. Funciona sin GPU: probado con llvmpipe (Mesa) por EGL en Linux.

```bash
pip install -r requirements.txt
```

## Uso

```bash
# hoja de cuadros de prueba (rápida, baja resolución)
python3 render.py sheet sheet.jpg 960 3

# banda sonora
python3 audio.py                     # -> soundtrack.wav

# render final en 4K, un MP4 por plano
python3 render.py shots out 3840               # 4:3  (3840x2880)
ASPECT=0.5625 python3 render.py shots out 3840 # 16:9 (3840x2160)

# montaje final + subtítulos + hoja de cuadros
python3 assemble.py out PATCH.mp4 PATCH_framesheet.jpg
```

Para volver a renderizar un solo plano, pasa su nombre: `python3 render.py shots out 3840 07c_punch`.

Como referencia, en una máquina de 2 núcleos sin GPU cada cuadro en 4K tarda alrededor de 1 segundo, y el corto completo, unos 25 minutos.

## Participar

Es un proyecto abierto: puedes animar planos nuevos, mejorar los rigs o el sombreado, agregar personajes o cambiar la música. Abre un issue con tu idea o manda un pull request.

Algunas ideas para empezar:

- Polvo y humo con un aspecto más dibujado; hoy son esferas.
- Expresiones faciales y animación de la boca.
- Ciclos de carrera con mejor contacto de los pies con el suelo.
- Un modo de exportación vertical (9:16).

## Licencia

MIT. Mira el archivo [LICENSE](LICENSE).
